#include "infer.h"

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cerr << "usage: predict <path-to-image> <path-to-exported-script-module>\n";
    return -1;
  }

  std::string image_path = argv[1];
  std::string model_path = argv[2];

  int image_height = 1024;
  int image_width = 1024;

  cv::Mat image = cv::imread(image_path);
  std::shared_ptr<torch::jit::script::Module> model = read_model(model_path);
  for (int i = 0; i < 1000; ++i)
  {
    cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);

    std::string pred, prob;
    tie(pred, prob) = infer(image, image_height, image_width, model);
  }

  std::cout << "PREDICTION  : " << pred << std::endl;
  std::cout << "CONFIDENCE  : " << prob << std::endl;

  return 0;
}
