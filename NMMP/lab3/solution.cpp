#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#define pi 3.1415926
#define h1s pow(h1, 2)
#define h2s pow(h2, 2)
#define sins1 pow(sin(pi * h1 / (2 * l1)), 2)
#define sins2 pow(sin(pi * h2 / (2 * l2)), 2)

double sigma(double h1, double h2, double l1, double l2) { return (2 * h2s * sins1 + 2 * h1s * sins2) / (h1s + h2s); }
double param(double h1, double h2, double l1, double l2) {
    double sig = sigma(h1, h2, l1, l2);
    return 2 / (1 + sqrt(sig * (2 - sig)));
}
double psi1(double y) { return sin(pi * y); }
double psi2(double y) { return std::abs(sin(2 * pi * y)); }
double psi3(double x) { return -x * (x + 1); }
double f(double x, double y) { return cosh(pow(x, 2) * y); }

void Linspace(std::vector<double>& to_fillin, double start, double h) {
    int n = 1 / h;
    to_fillin.resize(n + 1);
    for (int i = 0; i < n + 1; ++i) {
        to_fillin[i] = start + i * h;
    }
}

void print_vec(const std::vector<double>& vec, const char* filename) {
    std::ofstream fout(filename);
    for (int i = 0; i < vec.size(); ++i) { fout << vec[i] << " "; }
    fout.close();
}

void print_matr(const std::vector<std::vector<double>>& matr, const char* filename) {
    std::ofstream fout(filename);
    for (int i = 0; i < matr.size(); ++i) {
        for (int j = 0; j < matr[0].size(); ++j) {            
            fout << matr[i][j] << " ";
        }
        fout << "\n";
    }
    fout.close();
}

bool Condition(const std::vector<std::vector<double>>& _U, const std::vector<std::vector<double>>& _Ucopy, double _eps) {
    int m = _U.size(), n = _U[0].size();
    for (int i = 1; i < m - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            if (std::abs(_U[i][j] - _Ucopy[i][j]) > _eps) {
                return false;
            }
        }
    }
    return true;
}

void border(std::vector<std::vector<double>>& u, const std::vector<double>& x, const std::vector<double>& y) {
    for (int i = 0; i < x.size(); ++i) {
        u[0][i] = psi3(x[i]);
        u[y.size() - 1][i] = u[0][i];
    }
    for (int j = 1; j < y.size() - 1; ++j) {
        u[j][0] = psi1(y[j]);
        u[j][x.size() - 1] = psi2(y[j]);
    }
}

void Solution() {
    double h1 = 0.05, h2 = 0.1, a = -1, b = 0, c = 0, d = 1, eps = 1e-5,
            q = param(h1, h2, b - a, d - c);
    int N = 1 / h1, M = 1 / h2;

    std::vector<std::vector<double>> z(M + 1, std::vector<double>(N + 1, 0));
    std::vector<double> x, y;
    Linspace(x, a, h1);
    Linspace(y, c, h2);
    print_vec(x, "x.txt");
    print_vec(y, "y.txt");
    border(z, x, y);
    
    while (true) {
        std::vector<std::vector<double>> z_copy = z;
        for (int j = 1; j < M; ++j) {
            for (int i = 1; i < N; ++i) {
                z[j][i] = 1 / (2 / h1s + 2 / h2s) * ((z[j][i + 1] + z[j][i - 1]) / h1s + (z[j + 1][i] + z[j - 1][i]) / h2s + f(x[i], y[j])) + (1 - q) * z[j][i];
            }
        }
        if (Condition(z, z_copy, eps)) {
            break;
        }
    }
    print_matr(z, "z.txt");
}

int main() {
    Solution();

    return 0;
}