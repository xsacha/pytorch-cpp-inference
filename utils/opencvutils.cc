#include "opencvutils.h"

// Resize an image to a given size to
cv::Mat __resize_to_a_size(cv::Mat image, int new_height, int new_width) {
  // get image area and resized image area
  float img_area = float(image.rows * image.cols);
  float new_area = float(new_height * new_width);

  // resize
  cv::Mat image_scaled;
  cv::Size scale(new_width, new_height);

  if (new_area >= img_area) {
    cv::resize(image, image_scaled, scale, 0, 0, cv::INTER_LANCZOS4);
  } else {
    cv::resize(image, image_scaled, scale, 0, 0, cv::INTER_AREA);
  }

  return image_scaled;
}

// 1. Preprocess: Image must already be RGB
cv::Mat preprocess(cv::Mat image, int new_height, int new_width) {

  // Clone
  cv::Mat image_proc = image.clone();

  // Resize image
  if (new_height != 0 && new_width != 0)
  {
    image_proc = __resize_to_a_size(image_proc, new_height, new_width);
  }

  // Convert image to float
  image_proc.convertTo(image_proc, CV_32FC3);

  // 3. Normalize to [0, 1]
  image_proc = image_proc / 255.0;

  return image_proc;
}
