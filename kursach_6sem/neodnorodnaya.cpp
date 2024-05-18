#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#define pi 3.14159265358979

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

void print(const char* _fileName, vector<double> _vec) {
    ofstream fout(_fileName);
    for (size_t i = 0; i < _vec.size(); ++i) { fout << _vec[i] << " "; }
    fout.close();
}

double f(double _r, double _z, double _r0, double _z0, int _n = 20) {
    return _n / sqrt(pi) * exp(- _n * _n * (pow(_r - _r0, 2) + pow(_z - _z0, 2)));
}

void Solution() {
    ios_base::sync_with_stdio(0);
    cout.tie(0);

    int n = 201, m = 401;
    double R = 25, u0 = 5, a = 0, b = 200, c = -200, d = -c,
            eps = 1e-5;

    vector<double> r(n), z(m);
    Linspace(a, b, r);
    Linspace(c, d, z);
    print("r_n.txt", r);
    print("z_n.txt", z);

    double h_r = r[1] - r[0], h_z = z[1] - z[0], _1h_r_2 = 1 / pow (h_r, 2), _1h_z_2 = 1 / pow (h_z, 2);

    vector<vector<double>> u(m, vector<double>(n, 0));

    int k = 0, mid = (m - 1) / 2;

    while (r[k] <= R) {
        u[mid][k] = u0;
        ++k;
    }

    int z0 = m - 1 - 3 * mid / 4;

    int counter = 1;

    while (true) {
        vector<vector<double>> u_copy = u;

        for (int j = 1; j < m - 1; ++j) {
            if (j == mid) {
                for (int i = k; i < n - 1; ++i) {
                    u[j][i] = 1 / (2 * _1h_r_2 + 2 * _1h_z_2) * ((i + 1./2)*_1h_r_2/i * u[j][i + 1] + (i - 1./2)*_1h_r_2/i * u[j][i - 1] + _1h_z_2 * (u[j - 1][i] + u[j + 1][i]) + f(r[i], z[j], h_r, z[z0]));
                }
                continue;
            }
            for (int i = 1; i < n - 1; ++i) {
                u[j][i] = 1 / (2 * _1h_r_2 + 2 * _1h_z_2) * ((i + 1./2)*_1h_r_2/i * u[j][i + 1] + (i - 1./2)*_1h_r_2/i * u[j][i - 1] + _1h_z_2 * (u[j - 1][i] + u[j + 1][i]) + f(r[i], z[j], h_r, z[z0]));
            }
            u[j][0] = 4 * u[j][1] / 3 - u[j][2] / 3;
            //u[j][0] = u[j][1];
        }

        if (Condition(u, u_copy, eps)) {
            cout << "Number of iterations: " << counter << "\n";
            break;
        }
        if (counter % 1000 == 0) cout << counter << "\n";
        counter++;
    }

    ofstream fR("R_n.txt");
    fR << R;
    fR.close();

    ofstream fu("u_n.txt");
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