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

void approx1(vector<vector<double>>& u, const vector<double>& z, const vector<double>& r, double R, double u0) {
    double l = .04;
    for (int i = 0; i < u.size(); ++i) {
        for (int j = 0; j < u[i].size(); ++j) {
            if (r[j] <= R) {
                u[i][j] = u0 * exp(-l * abs(z[i]));
            } else {
                u[i][j] = u0 * exp(-l * sqrt(pow(r[j] - R, 2) + pow(z[i], 2)));
            }
        }
    }
}

void approx2(vector<vector<double>>& u, const vector<double>& z, const vector<double>& r, double R, double u0) {
    approx1(u, z, r, R, u0);

    for (int j = 0; j < u[0].size(); ++j) {
        u[0][j] = 2 * u0 * R / (pi * sqrt(pow(r[j], 2) + pow(z[0], 2)));
        u[u.size() - 1][j] = 2 * u0 * R / (pi * sqrt(pow(r[j], 2) + pow(z[u.size() - 1], 2)));
    }
    for (int i = 1; i < u.size() - 1; ++i) {
        u[i][u[i].size() - 1] = 2 * u0 * R / (pi * sqrt(pow(r[u[i].size() - 1], 2) + pow(z[i], 2)));
    }
}

void approx3(vector<vector<double>>& u, const vector<double>& z, const vector<double>& r, double R, double u0, int mid) {
    
    for (int j = 0; j < u[0].size(); ++j) {
        u[0][j] = 2 * u0 * R / (pi * sqrt(pow(r[j], 2) + pow(z[0], 2)));
        u[u.size() - 1][j] = 2 * u0 * R / (pi * sqrt(pow(r[j], 2) + pow(z[u.size() - 1], 2)));
    }
    for (int i = 1; i < u.size() - 1; ++i) {
        u[i][u[i].size() - 1] = 2 * u0 * R / (pi * sqrt(pow(r[u[i].size() - 1], 2) + pow(z[i], 2)));
    }
    
    int last_r = r.size() - 1, last_z = z.size() - 1;
    double l = (r[last_r] - R) / (10 * (z[last_z] - z[0]));
    for (int j = 0; j < u[mid].size() - 1; ++j) {
        if (r[j] <= R) {
            u[mid][j] = u0;
        } else {
            u[mid][j] = u0 * exp(-l * (r[j] - R)) + (u[mid][last_r] - u0 * exp(-l * (r[last_r] - R))) / sqrt(r[last_r] - R) * sqrt(r[j] - R);
        }
    }

    for (int i = 1; i < u.size() - 1; ++i) {
        if (i == mid) continue;

        for (int j = 0; j < u[i].size() - 1; ++j) {
            u[i][j] = u[mid][j] * exp(-l * abs(z[i])) + (u[last_z][j] - u[mid][j] * exp(-l * (z[last_z]))) / sqrt(z[last_z]) * sqrt(abs(z[i]));
        }
    }
}

double Q(const vector<vector<double>>& u, const vector<double>& r, double hz, int ind, int k) { // ind - индекс строки с диском, k - индекс элемента после последнего в диске
    double integ1 = 0, integ2 = 0, hr = r[1] - r[0];

    for (int i = 0; i < k - 1; ++i) {
        integ1 += (-(u[ind + 1][i] - u[ind][i]) / hz * r[i] - (u[ind + 1][i + 1] - u[ind][i + 1]) / hz * r[i + 1]);
        integ2 += (-(u[ind - 1][i] - u[ind][i]) / hz * r[i] - (u[ind - 1][i + 1] - u[ind][i + 1]) / hz * r[i + 1]);
    }
    integ1 *= (pi * hr);
    integ2 *= (pi * hr);

    return integ1 + integ2;
}

void Solution() {
    
    int n = 501, m = 1001;
    double R = 5, u0 = 5, a = 0, b = 10, c = -10, d = -c,
            eps = 1e-5;

    vector<double> r(n), z(m);
    Linspace(a, b, r);
    Linspace(c, d, z);
    print_vector("output/r_o.txt", r);
    print_vector("output/z_o.txt", z);
    print_scalar("output/R_o.txt", R);
    print_scalar("output/u0_o.txt", u0);

    double h_r = r[1] - r[0], h_z = z[1] - z[0], _1h_r_2 = 1 / pow (h_r, 2), _1h_z_2 = 1 / pow (h_z, 2)/*rel_sq = pow(h_r / h_z, 2), mul = 1 / (2 + 2 * rel_sq)*/;

    vector<vector<double>> u(m, vector<double>(n, 0));

    auto start = chrono::high_resolution_clock::now();

    int k = 0, mid = (m - 1) / 2;
    while (r[k] <= R) {
        u[mid][k] = u0;
        ++k;
    }

    approx2(u, z, r, R, u0);

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

        /*for (int j = 1; j < m - 1; ++j) {
            if (j == mid) {
                for (int i = k; i < n - 1; ++i) {
                    u[j][i] = 1 / (2 * _1h_r_2 + 2 * _1h_z_2) * ((i + 1./2)*_1h_r_2/i * u[j][i + 1] + (i - 1./2)*_1h_r_2/i * u[j][i - 1] + _1h_z_2 * (u[j - 1][i] + u[j + 1][i]));
                }
                continue;
            }
            for (int i = 1; i < n - 1; ++i) {
                u[j][i] = 1 / (2 * _1h_r_2 + 2 * _1h_z_2) * ((i + 1./2)*_1h_r_2/i * u[j][i + 1] + (i - 1./2)*_1h_r_2/i * u[j][i - 1] + _1h_z_2 * (u[j - 1][i] + u[j + 1][i]));
            }
            u[j][0] = 4 * u[j][1] / 3 - u[j][2] / 3;
            //u[j][0] = u[j][1];
        }*/

        /*NEW*/
        for (int i = k; i < n - 1; ++i) {
            u[mid][i] = 1 / (2 * _1h_r_2 + 2 * _1h_z_2) * ((i + 1./2)*_1h_r_2/i * u[mid][i + 1] + (i - 1./2)*_1h_r_2/i * u[mid][i - 1] + _1h_z_2 * (u[mid - 1][i] + u[mid + 1][i]));
        }

        for (int j = 1; j < mid; ++j) {
            for (int i = 1; i < n - 1; ++i) {
                u[mid + j][i] = 1 / (2 * _1h_r_2 + 2 * _1h_z_2) * ((i + 1./2)*_1h_r_2/i * u[mid + j][i + 1] + (i - 1./2)*_1h_r_2/i * u[mid + j][i - 1] + _1h_z_2 * (u[mid + j - 1][i] + u[mid + j + 1][i]));
                u[mid - j][i] = 1 / (2 * _1h_r_2 + 2 * _1h_z_2) * ((i + 1./2)*_1h_r_2/i * u[mid - j][i + 1] + (i - 1./2)*_1h_r_2/i * u[mid - j][i - 1] + _1h_z_2 * (u[mid - j - 1][i] + u[mid - j + 1][i]));
            }
            u[mid + j][0] = 4 * u[mid + j][1] / 3 - u[mid + j][2] / 3;
            u[mid - j][0] = 4 * u[mid - j][1] / 3 - u[mid - j][2] / 3;
        }
        /*NEW*/

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

    ofstream fu("output/u_o.txt");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            fu << u[i][j] << " ";
        }
        fu << "\n";
    }
    fu.close();

    double Q1 = Q(u, r, h_z, mid, k);
    
    cout << "Recounted for first disk: " << Q1 << ", was for first disk: " << 8 * u0 * R << "\n";
}

int main() {
    Solution();

    return 0;
}