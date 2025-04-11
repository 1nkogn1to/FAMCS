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

// Q = 2pi * \int_0^R \rho * \sigma(\rho) d\rho = 2pi * \sum 
double Q(const vector<vector<double>>& u, const vector<double>& r, double hz, int ind, int k) { // ind - индекс строки с диском, k - индекс элемента после последнего в диске
    double integ1 = 0, integ2 = 0, hr = r[1] - r[0];

    for (int i = 0; i < k - 1; ++i) {
        integ1 += (-(u[ind + 1][i] - u[ind][i]) / hz * r[i] - (u[ind + 1][i + 1] - u[ind][i + 1]) / hz * r[i + 1]);
        integ2 += (-(u[ind - 1][i] - u[ind][i]) / hz * r[i] - (u[ind - 1][i + 1] - u[ind][i + 1]) / hz * r[i + 1]);
    }
    integ1 *= (pi * hr);
    integ2 *= (pi * hr);

    return (integ1 + integ2) / 2;
}

void Solution() {
    
    int n = 101, m = 201;
    
    int dist = 35; // расстояние от диска до центра системы (оба диска равноудалены от центра)
    double R1 = 15, R2 = 40, u1 = 2, u2 = 1, a = 0, b = 50, c = -50, d = -c, eps = 1e-5;
 
    vector<double> r(n), z(m);
    Linspace(a, b, r);
    Linspace(c, d, z);
    print_vector("output/r.txt", r);
    print_vector("output/z.txt", z);

    double h_r = r[1] - r[0], h_z = z[1] - z[0], _1h_r_2 = 1 / pow (h_r, 2), _1h_z_2 = 1 / pow (h_z, 2)/*rel_sq = pow(h_r / h_z, 2), mul = 1 / (2 + 2 * rel_sq)*/, distance = h_z * 2 * dist;

    vector<vector<double>> u(m, vector<double>(n, 0));

    /** ЗАПУСК ТАЙМЕРА **/
    auto start = chrono::high_resolution_clock::now();
    
    /** заполнение дисков **/
    int k1 = 0, k2 = 0, mid = (m - 1) / 2;
    while (r[k1] <= R1) {
        u[mid - dist][k1] = u1;
        ++k1;
    }
    while (r[k2] <= R2) {
        u[mid + dist][k2] = u2;
        ++k2;
    }

    cout << z[mid - dist] << " " << z[mid + dist] << "\n";

    /*Probable approximation*/
    // code ...
    //approx1(u, r, z, u1, u2, R1, R2, mid, dist);

    /*ofstream fapp("output/approx.txt");
    for (int i = 0; i < u.size(); ++i) {
        for (int j = 0; j < u[i].size(); ++j) {
            fapp << u[i][j] << " ";
        }
        fapp << "\n";
    }
    fapp.close();*/

    int counter = 1;

    while (true) {
        vector<vector<double>> u_copy = u;

        for (int j = 1; j < m - 1; ++j) {
            int start = 1;
            if (j == mid - dist) {
                start = k1;
            }

            if (j == mid + dist) {
                start = k2;
            }

            for (int i = start; i < n - 1; ++i) {
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

    double Q1 = Q(u, r, h_z, mid - dist, k1),
        Q2 = Q(u, r, h_z, mid + dist, k2);
    
    cout << "Recounted for first disk: " << Q1 << ", was for first disk: " << 8 * u1 * R1 << "\n";
    cout << "Recounted for second disk: " << Q2 << ", was for second disk: " << 8 * u2 * R2 << "\n";
}

int main() {
    Solution();

    return 0;
}