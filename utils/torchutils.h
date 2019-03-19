#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>

#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

std::shared_ptr<torch::jit::script::Module> read_model(std::string);
std::vector<torch::Tensor> forward(cv::Mat, std::shared_ptr<torch::jit::script::Module>);
std::vector<float> as_float(torch::Tensor o);
std::vector<long> as_long(torch::Tensor o);
std::tuple<std::string, std::string> postprocess(std::vector<float>, std::vector<std::string>);
