#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#define pi 3.141592653589793

using namespace std;

bool condition(const vector<double>& _v1, const vector<double>& _v2, const double& _eps) {
    if (_v1.size() != _v2.size()) {
        cerr << "Error!";
        return false;
    }

    for (int i = 0; i < _v1.size(); ++i) {
        if (abs(_v1[i] - _v2[i]) > _eps) {
            return false;
        }
    }
    return true;
}

vector<double> SolveSystem(vector<vector<double>> _s, vector<double> _b, double _eps = 1e-6) {
    vector<double> x = _b; // начальное приближение
    int n = x.size();

    for (int j = 0; j < n - 1; j++) {
        for (int i = j + 1; i < n; i++) {
            double c = abs(_s[j][j]), s = abs(_s[i][j]);
            for (int k = j; k < n; k++) {
                double temp = _s[j][k];
                _s[j][k] = c * _s[j][k] + s * _s[i][k];
                _s[i][k] = -s * temp + c * _s[i][k];
            }
            double temp = _b[j];
            _b[j] = c * _b[j] + s * _b[i];
            _b[i] = -s * temp + c * _b[i];
        }
    }
    
    for (int i = n - 1; i > 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            _b[j] -= (_b[i] * _s[j][i] / _s[i][i]);
            _s[j][i] = 0;
        }
        _b[i] /= _s[i][i];
        _s[i][i] = 1;
    }
    _b[0] /= _s[0][0];
    _s[0][0] = 1;
    
    for (int i = 0; i < n; ++i) {
        x[i] =_b[i];
    }

    return x;
}

// проверка построенной формулы
double f(double x) {
    return pow(x, 3);
}

double quadrature_formula(vector<double> _A, vector<double> _x) {
    if (_A.size() != _x.size()) {
        return INFINITY;
    }
    double sum = 0;
    for (int i = 0; i < _x.size(); ++i) {
        sum += _A[i] * f(_x[i]);
    }
    return sum;
}

void Solution() {
    vector<double> vb = {pi, 5*pi/2, 51*pi/8/*, 265*pi/16, 5603*pi/128, 30075*pi/256*/};
    int n = vb.size();
    double a = 2, b = 3, h = (b - a) / (n - 1);
    vector<vector<double>> s(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            s[i][j] = pow(a + j*h, i);
            cout << s[i][j] << " ";
        }
        cout << "\n";
    }
    vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = a + i * h;
    }
    vector<double> A = SolveSystem(s, vb);
    for (int i = 0; i < n; ++i) {
        cout << setprecision(15) << A[i] << " ";
    }
    cout << quadrature_formula(A, x);
}

int main() {
    Solution();

    return 0;
}