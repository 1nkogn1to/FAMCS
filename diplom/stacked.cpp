#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#define pi 3.1415926

using namespace std;

void Linspace(double _start, double _end, vector<double>& _vec) {
    int n = _vec.size();
    double h = (_end - _start) / (n - 1.);

    for (int i = 0; i < n; ++i) { _vec[i] = _start + i * h; }
}

bool Condition(vector<vector<double>> _U, vector<vector<double>> _U_copy, double _eps) {
    int m = _U.size(), n = _U[0].size();
    for (int i = 1; i < m - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            if (abs(_U[i][j] - _U_copy[i][j]) > _eps) {
                return false;
            }
        }
    }
    return true;
}

void print_vector(const char* fileName, vector<double> _vec) {
    ofstream fout(fileName);
    for (size_t i = 0; i < _vec.size(); ++i) { fout << _vec[i] << " "; }
    fout.close();
}

void print_scalar(const char* fileName, double scal) {
    ofstream fout(fileName);
    fout << scal;
    fout.close();
}

void approx1(vector<vector<double>>& u, const vector<double>& r, const vector<double> z, double u1, double u2, double R1, double R2, int mid, int dist) {
    int fz = 0, lz = z.size() - 1, lr = r.size() - 1;
    for (int j = 0; j < r.size(); ++j) {
        u[fz][j] = 2 * u1 * R1 / (pi * sqrt(pow(r[j], 2) + pow(z[mid - dist] - z[fz], 2))) + 2 * u2 * R2 / (pi * sqrt(pow(r[j], 2) + pow(z[mid + dist] - z[fz], 2)));
        u[lz][j] = 2 * u1 * R1 / (pi * sqrt(pow(r[j], 2) + pow(z[lz] - z[mid - dist], 2))) + 2 * u2 * R2 / (pi * sqrt(pow(r[j], 2) + pow(z[lz] - z[mid + dist], 2)));
    }

    for (int i = 1; i < lz; ++i) {
        u[i][lr] = 2 * u1 * R1 / (pi * sqrt(pow(r[lr], 2) + pow(z[i] - z[mid - dist], 2))) + 2 * u2 * R2 / (pi * sqrt(pow(r[lr], 2) + pow(z[i] - z[mid + dist], 2)));
    }
}

void Solution() {
    
    int n = 51, m = 101;
    
    int distance_to_center_mul = 15;
    double R1 = 15, R2 = 25, u1 = 5, u2 = 5, a = 0, b = 50, c = -50, d = -c, eps = 1e-9;
 
    vector<double> r(n), z(m);
    Linspace(a, b, r);
    Linspace(c, d, z);
    print_vector("output/r.txt", r);
    print_vector("output/z.txt", z);
    print_scalar("output/dist.txt", distance_to_center_mul);

    double h_r = r[1] - r[0], h_z = z[1] - z[0], _1h_r_2 = 1 / pow (h_r, 2), _1h_z_2 = 1 / pow (h_z, 2)/*rel_sq = pow(h_r / h_z, 2), mul = 1 / (2 + 2 * rel_sq)*/, distance = h_z * 2 * distance_to_center_mul;

    vector<vector<double>> u(m, vector<double>(n, 0));

    /** ЗАПУСК ТАЙМЕРА **/
    auto start = chrono::high_resolution_clock::now();
    
    /** заполнение дисков **/
    int k1 = 0, k2 = 0, mid = (m - 1) / 2;
    while (r[k1] <= R1) {
        u[mid - distance_to_center_mul][k1] = u1;
        ++k1;
    }
    while (r[k2] <= R2) {
        u[mid + distance_to_center_mul][k2] = u2;
        ++k2;
    }

    cout << z[mid - distance_to_center_mul] << " " << z[mid + distance_to_center_mul] << "\n";

    /*Probable approximation*/
    // code ...
    approx1(u, r, z, u1, u2, R1, R2, mid, distance_to_center_mul);

    ofstream fapp("output/approx.txt");
    for (int i = 0; i < u.size(); ++i) {
        for (int j = 0; j < u[i].size(); ++j) {
            fapp << u[i][j] << " ";
        }
        fapp << "\n";
    }
    fapp.close();

    int counter = 1;

    while (true) {
        vector<vector<double>> u_copy = u;

        for (int j = 1; j < m - 1; ++j) {
            if (j == mid - distance_to_center_mul) {
                for (int i = k1; i < n - 1; ++i) {
                    u[j][i] = 1 / (2 * _1h_r_2 + 2 * _1h_z_2) * ((i + 1./2)*_1h_r_2/i * u[j][i + 1] + (i - 1./2)*_1h_r_2/i * u[j][i - 1] + _1h_z_2 * (u[j - 1][i] + u[j + 1][i]));
                }
                continue;
            }

            if (j == mid + distance_to_center_mul) {
                for (int i = k2; i < n - 1; ++i) {
                    u[j][i] = 1 / (2 * _1h_r_2 + 2 * _1h_z_2) * ((i + 1./2)*_1h_r_2/i * u[j][i + 1] + (i - 1./2)*_1h_r_2/i * u[j][i - 1] + _1h_z_2 * (u[j - 1][i] + u[j + 1][i]));
                }
                continue;
            }

            for (int i = 1; i < n - 1; ++i) {
                u[j][i] = 1 / (2 * _1h_r_2 + 2 * _1h_z_2) * ((i + 1./2)*_1h_r_2/i * u[j][i + 1] + (i - 1./2)*_1h_r_2/i * u[j][i - 1] + _1h_z_2 * (u[j - 1][i] + u[j + 1][i]));
            }
            u[j][0] = 4 * u[j][1] / 3 - u[j][2] / 3;
        }

        /*Probable optimization*/
        // code ...

        if (Condition(u, u_copy, eps)) {
            cout << "Number of iterations: " << counter << "\n";
            break;
        }
        // if (counter % 1000 == 0) cout << counter << "\n";
        counter++;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Время выполнения (сек): " << duration.count() << "\n";

    ofstream fu("output/u.txt");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            fu << u[i][j] << " ";
        }
        fu << "\n";
    }
    fu.close();
}

int main() {
    Solution();

    return 0;
}