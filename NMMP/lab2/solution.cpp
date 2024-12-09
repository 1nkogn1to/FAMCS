#include <iostream>
#include <vector>
#include <cmath>
#define tau2 pow(tau, 2)

void Solution(std::vector<std::vector<double>>& u, double tau = 0.1) {
    double ax = 0, bx = 1, at = 0, bt = 1, h = 0.1;
    int N = int((bx - ax) / h), M = int((bt - at) / tau);
    std::vector<double> x(N + 1);
    std::vector<double> t(M + 1);
    //std::cout << M + 1 << " " << N + 1 << "\n";

    for (int i = 0; i < N + 1; ++i) { x[i] = i * h; }
    for (int j = 0; j < M + 1; ++j) { t[j] = j * tau; }

    u.resize(M + 1, std::vector<double>(N + 1, 0));

    for (int i = 0; i < N + 1; ++i) { u[0][i] = pow(x[i], 2); }

    u[1][0] = pow(t[1], 3);
    for (int i = 1; i < N + 1; ++i) {
        u[1][i] = u[0][i];
    }

    for (int j = 2; j < M + 1; ++j) {
        u[j][0] = pow(t[j], 3);
        for (int i = 1; i < N; ++i) {
            u[j][i] = 2 * u[j - 1][i] - u[j - 2][i] + pow(tau, 2) * ((u[j - 1][i - 1] - 2 * u[j - 1][i] + u[j - 1][i + 1]) / pow(h, 2) + 6 * t[j] - 2);
        }
        u[j][N] = (2 - 2 * tau2 - 2 * tau2 / h) * u[j - 1][N] + 2 * tau2 * u[j - 1][N - 1] - u[j - 2][N] + 2 * tau2 * (3 + pow(t[j - 1], 3)) / h + 6 * tau2 * t[j - 1] - 2 * tau2;
    }
}

void print_matrix(const char* filename, const std::vector<std::vector<double>>& u) {
    freopen(filename, "w", stdout);
    for (int j = 0; j < u.size(); ++j) {
        for (int i = 0; i < u[j].size(); ++i) {
            std::cout << u[j][i] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    std::vector<std::vector<double>> u1, u01, u2;

    Solution(u1);
    Solution(u01, 0.01);
    Solution(u2, 0.2); // должно получиться кривое решение

    print_matrix("output1.txt", u1);
    print_matrix("output01.txt", u01);
    print_matrix("output2.txt", u2);

    return 0;
}