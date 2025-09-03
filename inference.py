import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, 'third_part')
sys.path.insert(0, 'third_part/GPEN')
sys.path.insert(0, 'third_part/GFPGAN')

# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from third_part.GFPGAN.gfpgan import GFPGANer
# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, options, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict, estimate_pose_from_landmarks, \
                                  calculate_pose_confidence_score, apply_schmitt_trigger, apply_state_machine, fix_boundary_flickers, apply_temporal_smoothing
import warnings
warnings.filterwarnings("ignore")

args = options()

def main():    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[Info] Using {} for inference.'.format(device))
    os.makedirs(os.path.join('temp', args.tmp_dir), exist_ok=True)

    enhancer = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False, \
                               sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
    restorer = GFPGANer(model_path='checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean', \
                        channel_multiplier=2, bg_upsampler=None)

    base_name = args.face.split('/')[-1]
    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')
    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print ("[Step 0] Number of frames available for inference: "+str(len(full_frames)))
    # face detection & cropping, cropping the first frame as the style of FFHQ
    croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
    full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]
    crop_result = croper.crop(full_frames_RGB, xsize=512)

    # Handle case where no face can be detected in the video
    if crop_result is None:
        print("[WARNING] No face could be detected in the video. Using original video with new audio.")
        print("[INFO] Bypassing all ML processing and directly combining original video with target audio.")
        
        # Create output directory if it doesn't exist
        outfile_dir = os.path.dirname(args.outfile)
        if outfile_dir:
            os.makedirs(outfile_dir, exist_ok=True)
        
        # Use ffmpeg to combine original video with new audio
        temp_video_path = 'temp/original_video_no_audio.mp4'
        
        # Extract video without audio
        cmd = ['ffmpeg', '-y', '-i', args.face, '-an', '-c:v', 'copy', temp_video_path]
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Combine with new audio
        cmd = ['ffmpeg', '-y', '-i', temp_video_path, '-i', args.audio, 
               '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', args.outfile]
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        print(f"[INFO] Result saved to {args.outfile}")
        return

    full_frames_RGB, crop, quad = crop_result
    clx, cly, crx, cry = crop
    lx, ly, rx, ry = quad
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly+ly, min(cly+ry, full_frames[0].shape[0]), clx+lx, min(clx+rx, full_frames[0].shape[1])
    # original_size = (ox2 - ox1, oy2 - oy1)
    frames_pil = [Image.fromarray(cv2.resize(frame,(256,256))) for frame in full_frames_RGB]

    # get the landmark according to the detected face.
    if not os.path.isfile('temp/'+base_name+'_landmarks.txt') or args.re_preprocess:
        print('[Step 1] Landmarks Extraction in Video.')
        kp_extractor = KeypointExtractor()
        lm = kp_extractor.extract_keypoint(frames_pil, './temp/'+base_name+'_landmarks.txt')
    else:
        print('[Step 1] Using saved landmarks.')
        lm = np.loadtxt('temp/'+base_name+'_landmarks.txt').astype(np.float32)
        lm = lm.reshape([len(full_frames), -1, 2])
       
    if not os.path.isfile('temp/'+base_name+'_coeffs.npy') or args.exp_img is not None or args.re_preprocess:
        net_recon = load_face3d_net(args.face3d_net_path, device)
        lm3d_std = load_lm3d('checkpoints/BFM')

        video_coeffs = []
        
        # Stage 1: Collect raw confidence scores
        raw_confidence_scores = []
        smoothed_confidence_scores = []
        ema_alpha = 0.6  # Exponential moving average smoothing factor - increased for more responsiveness
        
        print(f"[Stage 1] Collecting raw pose confidence scores...")
        
        for idx in tqdm(range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
            frame = frames_pil[idx]
            W, H = frame.size
            lm_idx = lm[idx].reshape([-1, 2])
            
            landmark_detection_failed = (np.mean(lm_idx) == -1)
            if landmark_detection_failed:
                # The rest of the fallback logic remains to ensure coeffs array has consistent shape
                lm_idx = (lm3d_std[:, :2]+1) / 2.
                lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
            else:
                lm_idx[:, -1] = H - 1 - lm_idx[:, -1]


            trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, lm3d_std)
            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0) 
            with torch.no_grad():
                coeffs = split_coeff(net_recon(im_idx_tensor))

            # --- Stage 1: Calculate Raw Confidence Score ---
            if landmark_detection_failed:
                # No face detected - very low confidence
                raw_confidence = 0.0
                pitch_3dmm_deg, yaw_3dmm_deg = 0.0, 0.0
                lm_confidence = 0.0
                print(f"Frame {idx}: No landmarks - raw_confidence=0.0")
            else:
                # Method 1: 3DMM-based angle detection
                angles = coeffs['angle'].cpu().numpy()[0] # [pitch, yaw, roll]
                pitch_3dmm, yaw_3dmm, _ = angles
                pitch_threshold = getattr(args, 'pose_pitch_threshold', args.pose_angle_threshold)
                yaw_threshold = getattr(args, 'pose_yaw_threshold', args.pose_angle_threshold)
                
                # Convert from radians to degrees
                if abs(pitch_3dmm) < 6.28 and abs(yaw_3dmm) < 6.28:  # Likely radians
                    pitch_3dmm_deg = np.degrees(pitch_3dmm)
                    yaw_3dmm_deg = np.degrees(yaw_3dmm)
                else:  # Already in degrees
                    pitch_3dmm_deg = pitch_3dmm
                    yaw_3dmm_deg = yaw_3dmm
                
                # Method 2: Landmark-based pose detection with confidence
                lm_current = lm[idx].reshape([-1, 2])
                pitch_lm_deg, yaw_lm_deg, lm_confidence = estimate_pose_from_landmarks(lm_current)
                
                # Calculate unified confidence score using OR logic
                raw_confidence = calculate_pose_confidence_score(
                    pitch_3dmm_deg, yaw_3dmm_deg, lm_confidence, 
                    pitch_threshold, yaw_threshold
                )
                
                # Additional check: if our landmark-based pose detection finds extreme angles, override
                if pitch_lm_deg is not None and yaw_lm_deg is not None:
                    if abs(pitch_lm_deg) >= pitch_threshold or abs(yaw_lm_deg) >= yaw_threshold:
                        raw_confidence = 0.2  # Override to low confidence if landmarks detect extreme pose
                        print(f"Frame {idx}: Landmark override - pitch_lm={pitch_lm_deg:.1f}°, yaw_lm={yaw_lm_deg:.1f}°")
                
                print(f"Frame {idx}: 3DMM: pitch={pitch_3dmm_deg:.1f}°, yaw={yaw_3dmm_deg:.1f}° | LM: conf={lm_confidence:.2f} | Raw score: {raw_confidence:.3f}")
            
            raw_confidence_scores.append(raw_confidence)
            
            # Apply exponential moving average smoothing
            if idx == 0:
                smoothed_score = raw_confidence
            else:
                smoothed_score = ema_alpha * raw_confidence + (1 - ema_alpha) * smoothed_confidence_scores[-1]
            smoothed_confidence_scores.append(smoothed_score)

            pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],\
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        semantic_npy = np.array(video_coeffs)[:,0]
        np.save('temp/'+base_name+'_coeffs.npy', semantic_npy)
        
        # --- Simple Pose Viability Logic ---
        print(f"\n[Stage 2] Applying simple pose viability logic...")
        print(f"Raw confidence scores: min={min(raw_confidence_scores):.3f}, max={max(raw_confidence_scores):.3f}, mean={np.mean(raw_confidence_scores):.3f}")
        
        # Simple threshold-based decision
        threshold = 0.4  # Simple threshold: below this = not viable
        pose_is_viable_raw = [score >= threshold for score in raw_confidence_scores]
        
        print(f"Simple threshold (0.4): {sum(pose_is_viable_raw)}/{len(pose_is_viable_raw)} frames marked viable")
        
        # Apply temporal smoothing: flip V islands of size <= n
        island_size = 2  # Start with flipping V islands of size 2 or less
        pose_is_viable = apply_temporal_smoothing(pose_is_viable_raw, island_size)
        
        print(f"After temporal smoothing (island_size={island_size}): {sum(pose_is_viable)}/{len(pose_is_viable)} frames marked viable")
        
        # Debug: Print final sequence pattern
        pattern = ''.join(['V' if v else 'N' for v in pose_is_viable])
        print(f"Final pattern: {pattern}")
        
        # Debug: Print frame-by-frame comparison
        print(f"\nFrame-by-frame comparison:")
        print("Frame | Raw   | Raw V/N| Final")
        print("------|-------|--------|------")
        for i in range(min(20, len(pose_is_viable))):  # Show first 20 frames
            print(f"{i:5d} | {raw_confidence_scores[i]:.3f} | {'V' if pose_is_viable_raw[i] else 'N':7s} | {'V' if pose_is_viable[i] else 'N':5s}")
        if len(pose_is_viable) > 20:
            print("...")
            for i in range(max(20, len(pose_is_viable)-10), len(pose_is_viable)):  # Show last 10 frames
                print(f"{i:5d} | {raw_confidence_scores[i]:.3f} | {'V' if pose_is_viable_raw[i] else 'N':7s} | {'V' if pose_is_viable[i] else 'N':5s}")
    else:
        print('[Step 2] Using saved coeffs.')
        semantic_npy = np.load('temp/'+base_name+'_coeffs.npy').astype(np.float32)
        # If using cached coeffs, we need to initialize pose_is_viable with all True
        # This means we accept all poses when using cached data
        pose_is_viable = [True] * len(semantic_npy)

    # generate the 3dmm coeff from a single image
    if args.exp_img is not None and ('.png' in args.exp_img or '.jpg' in args.exp_img):
        print('extract the exp from',args.exp_img)
        exp_pil = Image.open(args.exp_img).convert('RGB')
        lm3d_std = load_lm3d('third_part/face3d/BFM')
        
        W, H = exp_pil.size
        kp_extractor = KeypointExtractor()
        lm_exp = kp_extractor.extract_keypoint([exp_pil], 'temp/'+base_name+'_temp.txt')[0]
        if np.mean(lm_exp) == -1:
            lm_exp = (lm3d_std[:, :2] + 1) / 2.
            lm_exp = np.concatenate(
                [lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
        else:
            lm_exp[:, -1] = H - 1 - lm_exp[:, -1]

        trans_params, im_exp, lm_exp, _ = align_img(exp_pil, lm_exp, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_exp_tensor = torch.tensor(np.array(im_exp)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0)
        with torch.no_grad():
            expression = split_coeff(net_recon(im_exp_tensor))['exp'][0]
        del net_recon
    elif args.exp_img == 'smile':
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_mouth'])[0]
    else:
        print('using expression center')
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]

    # load DNet, model(LNet and ENet)
    D_Net, model = load_model(args, device)

    if not os.path.isfile('temp/'+base_name+'_stablized.npy') or args.re_preprocess:
        imgs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stabilize the expression In Video:"):
            if args.one_shot:
                source_img = trans_image(frames_pil[0]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[0:1]
            else:
                source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[idx:idx+1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
            coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(device)
        
            # hacking the new expression
            coeff[:, :64, :] = expression[None, :64, None].to(device) 
            with torch.no_grad():
                output = D_Net(source_img, coeff)
            img_stablized = np.uint8((output['fake_image'].squeeze(0).permute(1,2,0).cpu().clamp_(-1, 1).numpy() + 1 )/2. * 255)
            imgs.append(cv2.cvtColor(img_stablized,cv2.COLOR_RGB2BGR)) 
        np.save('temp/'+base_name+'_stablized.npy',imgs)
        del D_Net
    else:
        print('[Step 3] Using saved stabilized video.')
        imgs = np.load('temp/'+base_name+'_stablized.npy')
    torch.cuda.empty_cache()

    if not args.audio.endswith('.wav'):
        command = 'ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(args.audio, 'temp/{}/temp.wav'.format(args.tmp_dir))
        subprocess.call(command, shell=True)
        args.audio = 'temp/{}/temp.wav'.format(args.tmp_dir)
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    # --- New Silence Detection Logic ---
    is_speech_present = []
    wav_step_size = 16000 // int(fps) # samples per frame
    for i in range(0, len(wav), wav_step_size):
        chunk = wav[i : i + wav_step_size]
        if len(chunk) > 0:
            is_speech_present.append(not audio.is_silent(chunk))
    # --- End New Logic ---

    mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./fps, 0, []
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
    imgs = imgs[:len(mel_chunks)]
    full_frames = full_frames[:len(mel_chunks)]  
    lm = lm[:len(mel_chunks)]
    # Also truncate pose viability array and speech detection array to match audio length
    pose_is_viable = pose_is_viable[:len(mel_chunks)]
    is_speech_present = is_speech_present[:len(mel_chunks)]
    
    imgs_enhanced = []
    for idx in tqdm(range(len(imgs)), desc='[Step 5] Reference Enhancement'):
        img = imgs[idx]
        try:
            pred, _, _ = enhancer.process(img, img, face_enhance=True, possion_blending=False)
            imgs_enhanced.append(pred)
        except Exception as e:
            print(f"Warning: Face enhancement failed for frame {idx}, using original image. Error: {e}")
            imgs_enhanced.append(img)
    gen = datagen(imgs_enhanced.copy(), mel_chunks, full_frames, None, (oy1,oy2,ox1,ox2))

    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/{}/result.mp4'.format(args.tmp_dir), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    
    if args.up_face != 'original':
        instance = GANimationModel()
        instance.initialize()
        instance.setup()

    kp_extractor = KeypointExtractor()
    for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(tqdm(gen, desc='[Step 6] Lip Synthesis:', total=int(np.ceil(float(len(mel_chunks)) / args.LNet_batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(device)/255. # BGR -> RGB
        
        with torch.no_grad():
            incomplete, reference = torch.split(img_batch, 3, dim=1) 
            pred, low_res = model(mel_batch, img_batch, reference)
            pred = torch.clamp(pred, 0, 1)

            if args.up_face in ['sad', 'angry', 'surprise']:
                tar_aus = exp_aus_dict[args.up_face]
            else:
                pass
            
            if args.up_face == 'original':
                cur_gen_faces = img_original
            else:
                test_batch = {'src_img': torch.nn.functional.interpolate((img_original * 2 - 1), size=(128, 128), mode='bilinear'), 
                              'tar_aus': tar_aus.repeat(len(incomplete), 1)}
                instance.feed_batch(test_batch)
                instance.forward()
                cur_gen_faces = torch.nn.functional.interpolate(instance.fake_img / 2. + 0.5, size=(384, 384), mode='bilinear')
                
            if args.without_rl1 is not False:
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                mask = torch.where(incomplete==0, torch.ones_like(incomplete), torch.zeros_like(incomplete)) 
                pred = pred * mask + cur_gen_faces * (1 - mask) 
        
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        torch.cuda.empty_cache()
        
        batch_start_index = i * args.LNet_batch_size
        for j, (p, f, xf, c) in enumerate(zip(pred, frames, f_frames, coords)):
            frame_index = batch_start_index + j

            if not pose_is_viable[frame_index]:
                # If pose is not viable, write the original full frame and continue
                print(f"Using original frame for frame {frame_index} (pose not viable)")
                out.write(xf) 
                continue

            y1, y2, x1, x2 = c

            if is_speech_present[frame_index]:
                # Speech is present, use the normal lip-sync + enhancement pipeline
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                
                ff = xf.copy() 
                ff[y1:y2, x1:x2] = p
                
                # month region enhancement by GFPGAN
                cropped_faces, restored_faces, restored_img = restorer.enhance(
                    ff, has_aligned=False, only_center_face=True, paste_back=True)
                    # 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
                mm = [0,   0,   0,   0,   0,   0,   0,   0,   0,  0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
                mouse_mask = np.zeros_like(restored_img)
                tmp_mask = enhancer.faceparser.process(restored_img[y1:y2, x1:x2], mm)[0]
                mouse_mask[y1:y2, x1:x2]= cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.

                height, width = ff.shape[:2]
                restored_img, ff, full_mask = [cv2.resize(x, (512, 512)) for x in (restored_img, ff, np.float32(mouse_mask))]
                img = Laplacian_Pyramid_Blending_with_mask(restored_img, ff, full_mask[:, :, 0], 10)
                pp = np.uint8(cv2.resize(np.clip(img, 0 ,255), (width, height)))

                pp, orig_faces, enhanced_faces = enhancer.process(pp, xf, bbox=c, face_enhance=False, possion_blending=True)
            else:
                # Silence, use the stabilized frame without additional enhancement
                # `imgs` holds the stabilized frames from Step 3
                stabilized_frame = f # `f` is the stabilized frame from the generator
                pp = stabilized_frame

            out.write(pp)
    out.release()
    
    outfile_dir = os.path.dirname(args.outfile)
    if outfile_dir and not os.path.isdir(outfile_dir):
        os.makedirs(outfile_dir, exist_ok=True)
    command = 'ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/{}/result.mp4'.format(args.tmp_dir), args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')
    print('outfile:', args.outfile)


# frames:256x256, full_frames: original size
def datagen(frames, mels, full_frames, frames_pil, cox):
    img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch = [], [], [], [], [], []
    base_name = args.face.split('/')[-1]
    refs = []
    image_size = 256 

    # original frames
    kp_extractor = KeypointExtractor()
    fr_pil = [Image.fromarray(frame) for frame in frames]
    lms = kp_extractor.extract_keypoint(fr_pil, 'temp/'+base_name+'x12_landmarks.txt')
    frames_pil = [ (lm, frame) for frame,lm in zip(fr_pil, lms)] # frames is the croped version of modified face
    crops, orig_images, quads  = crop_faces(image_size, frames_pil, scale=1.0, use_fa=True)
    inverse_transforms = [calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]
    del kp_extractor.detector

    oy1,oy2,ox1,ox2 = cox
    face_det_results = face_detect(full_frames, args, jaw_correction=True)

    for inverse_transform, crop, full_frame, face_det in zip(inverse_transforms, crops, full_frames, face_det_results):
        imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
            cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))

        ff = full_frame.copy()
        ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
        
        # face_det is always a list [image, coords], even for dummy results
        oface, coords = face_det
        
        y1, y2, x1, x2 = coords
        refs.append(ff[y1: y2, x1:x2])

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face = refs[idx]
        
        # face_det_results[idx] is always a list [image, coords], even for dummy results
        oface, coords = face_det_results[idx]

        face = cv2.resize(face, (args.img_size, args.img_size))
        oface = cv2.resize(oface, (args.img_size, args.img_size))

        img_batch.append(oface)
        ref_batch.append(face) 
        mel_batch.append(m)
        coords_batch.append(coords)
        frame_batch.append(frame_to_save)
        full_frame_batch.append(full_frames[idx].copy())

        if len(img_batch) >= args.LNet_batch_size:
            img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
            img_masked = img_batch.copy()
            img_original = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
            img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, ref_batch  = [], [], [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
        img_masked = img_batch.copy()
        img_original = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch


if __name__ == '__main__':
    main()
