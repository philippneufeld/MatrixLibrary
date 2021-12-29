
#include <chrono>
#include <thread>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>
#include <ctime>
#include <chrono>

// #define ML_MATH_NO_INTRINSICS

#include <MatrixLibrary/Math/Matrix.h>
#include <blaze/Blaze.h>

using namespace ML;

template<typename MT> auto Rows(const MT& m) { return m.Rows(); }
template<std::size_t n>
auto Rows(const blaze::StaticMatrix<float, n, n>& m) { return m.rows(); }
template<typename MT> auto Cols(const MT& m) { return m.Cols(); }
template<std::size_t n>
auto Cols(const blaze::StaticMatrix<float, n, n>& m) { return m.columns(); }

template<typename T>
void printMatrix(const T& mat)
{
  for (size_t i = 0; i < Rows(mat); i++)
  {
    for (size_t j = 0; j < Cols(mat); j++)
    {
      std::cout << std::setw(6) << std::setprecision(3) << mat(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template<typename A, typename B, typename C>
void mmul_naive(C& c, const A& a, const B& b)
{
  c.SetZero();
  for (size_t i = 0; i < c.Rows(); i++)
  {
    for (size_t k = 0; k < a.Rows(); k++)
    {
      for (size_t j = 0; j < c.Cols(); j++)
      {
        c(i, j) += a(i, k) * b(k, j);
      }     
    }    
  }
}


template<typename MT, typename Func>
float benchmark(const std::string& name, Func func, double time_budget=1.0)
{
  double tmin = INFINITY;
  double tacc = 0;
  std::size_t runs = 0;
  auto ts_tot = std::chrono::high_resolution_clock::now(); 
  for (;(std::chrono::high_resolution_clock::now() - ts_tot).count() / 1e9 < time_budget; runs++)
  {
    MT mat1, mat2, res_mat;
    
    for (size_t i = 0; i < Rows(mat1); i++)
      for (size_t j = 0; j < Cols(mat1); j++)
        mat1(i, j) = float(rand()) / float(RAND_MAX);
    for (size_t i = 0; i < Rows(mat2); i++)
      for (size_t j = 0; j < Cols(mat2); j++)
        mat2(i, j) = float(rand()) / float(RAND_MAX);

    auto ts = std::chrono::high_resolution_clock::now();
    func(res_mat, mat1, mat2);
    double trun = (std::chrono::high_resolution_clock::now() - ts).count() / 1e9;

    tmin = std::min(tmin, trun);
    tacc += trun;

    if (time(NULL) == 5)
    {
      printMatrix(mat1);
      printMatrix(mat2);
      printMatrix(res_mat);
    }
  }
  double tav = tacc / runs;

  std::cout << name << std::endl;
  std::cout << tmin << "s (av: " << tav << "s)" << std::endl;

  auto n = Rows(MT());
  double ops = 2 * n*n*n;

  double flops_best = ops / tmin;
  double flops = ops / tav;
  std::cout << flops_best / 1e9 << " GFlops (av: " << flops / 1e9 << " GFlops)" << std::endl;

  double flops_max = 2 * 2 * 8 * 3.0e9;
  std::cout << flops_max / 1e9 << " GFlops (max)" << std::endl;

  double efficiency_best = flops_best / flops_max;
  double efficiency = flops / flops_max;
  std::cout << efficiency_best * 100 << "% (av: " << efficiency_best * 100 << "%)" << std::endl << std::endl;

  return efficiency;
}


#include <fstream>

int main()
{
  std::vector<double> mat_sizes, perf_native, perf_MLMath, perf_blaze;

  MLConstexprFor<std::size_t, 1, 4 + 1, 1>([&](auto i){
    auto budget = 0.2;
    mat_sizes.push_back(i);
    perf_native.push_back(benchmark<TMLStaticMatrix<float, i, i, true, 8>>("mmul_naive", [=](auto c, auto a, auto b) { mmul_naive(c, a, b); }, budget));
    perf_MLMath.push_back(benchmark<TMLStaticMatrix<float, i, i, true, 8>>("MLMath", [=](auto c, auto a, auto b) { c = a * b; }, budget));
    perf_blaze.push_back(benchmark<blaze::StaticMatrix<float, i, i>>("blaze", [=](auto c, auto a, auto b) { c = a * b; }, budget));
  });


  std::ofstream f;
  f.open("./performance.txt", std::ios::out);
  for(std::size_t i = 0; i < mat_sizes.size(); i++)
    f << mat_sizes[i] << "\t" << perf_native[i] << "\t" << perf_MLMath[i] << "\t" << perf_blaze[i] << std::endl;
  f.close();

  return 0;
}
