# This script converts the output test images into mp4 videos
import cv2
import os
image_path = '/mnt/IronWolf/logs/DiBS/2D/grid_sampling_256/snowflake/siren_origin/vis/' # path to image output directory
output_path = os.path.join(image_path, 'vids')
os.makedirs(output_path, exist_ok=True)
image_siz = 800
fps = 15
fourcc_encoder = cv2.VideoWriter_fourcc(*'mp4v')
extension = '.mp4'
out_sdf_eikonal_div = cv2.VideoWriter(os.path.join(output_path, 'sdf_eikonal_div' + extension), fourcc_encoder, fps, (image_siz*3, image_siz))
out_sdf_eikonal_div_zdiff = cv2.VideoWriter(os.path.join(output_path, 'sdf_eikonal_div_zdiff' + extension), fourcc_encoder, fps, (image_siz*4, image_siz))
out_sdf = cv2.VideoWriter(os.path.join(output_path, 'sdf' + extension), fourcc_encoder, fps, (image_siz, image_siz))
out_eikonal= cv2.VideoWriter(os.path.join(output_path, 'eikonal' + extension), fourcc_encoder, fps, (image_siz, image_siz))
out_div = cv2.VideoWriter(os.path.join(output_path, 'div' + extension), fourcc_encoder, fps, (image_siz, image_siz))
out_curl = cv2.VideoWriter(os.path.join(output_path, 'curl' + extension), fourcc_encoder, fps, (image_siz, image_siz))
out_zdiff = cv2.VideoWriter(os.path.join(output_path, 'zdiff' + extension), fourcc_encoder, fps, (image_siz, image_siz))

sdf_file_list = [file for file in os.listdir(image_path) if file.endswith('.png') and 'sdf' in file]
eikonal_file_list = [file for file in os.listdir(image_path) if file.endswith('.png') and 'eikonal' in file]
div_file_list = [file for file in os.listdir(image_path) if file.endswith('.png') and 'div' in file]
curl_file_list = [file for file in os.listdir(image_path) if file.endswith('.png') and 'curl' in file]
zdiff_file_list = [file for file in os.listdir(image_path) if file.endswith('.png') and 'zdiff' in file]
sdf_file_list.sort(); eikonal_file_list.sort(); div_file_list.sort(); curl_file_list.sort(); zdiff_file_list.sort()

n_files = sum('sdf' in s for s in sdf_file_list)
for i in range(n_files):
    sdf_img = cv2.imread(os.path.join(image_path, sdf_file_list[i]))
    eikonal_img = cv2.imread(os.path.join(image_path, eikonal_file_list[i]))
    div_img = cv2.imread(os.path.join(image_path, div_file_list[i]))
    curl_img = cv2.imread(os.path.join(image_path, curl_file_list[i]))
    zdiff_img = cv2.imread(os.path.join(image_path, zdiff_file_list[i]))
    out_sdf.write(sdf_img); out_eikonal.write(eikonal_img); out_div.write(div_img); out_curl.write(curl_img); out_zdiff.write(zdiff_img);
    out_sdf_eikonal_div.write(cv2.hconcat([sdf_img, eikonal_img, div_img]))
    out_sdf_eikonal_div_zdiff.write(cv2.hconcat([sdf_img, eikonal_img, div_img, zdiff_img]))

out_sdf_eikonal_div.release(); out_sdf.release(); out_eikonal.release(); out_div.release(); out_curl.release(); out_zdiff.release(); out_sdf_eikonal_div_zdiff.release();