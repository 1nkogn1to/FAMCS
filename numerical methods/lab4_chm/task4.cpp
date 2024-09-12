#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double ro(double x) {
    return sqrt(x) * exp(x);
}
double f(double x) {
    return sin(pow(x, 2)) / (1 + pow(log(x + 1), 2));
}

// Аппроксимация 4 производной, потому что аналитический вид функции очень огромный
vector<double> D4f(const vector<double>& _x) {
    int n = _x.size();
    double h = _x[1] - _x[0];

    vector<double> x_ext(n + 4);
    for (int i = 0; i < n + 4; ++i) {
        x_ext[i] = _x[0] + (i - 2) * h;
    }

    vector<double> d4f(n);
    for (int i = 0; i < n; ++i) {
        d4f[i] = (f(_x[i]) - 4*f(_x[i+1]) + 6*f(_x[i+2]) - 4*f(_x[i+3]) + f(_x[i+4])) / pow(h, 4);
    }
    cout << d4f[151] << "\n";
    cout << f(x_ext[153]) << "hui\n";
    x_ext.clear();

    return d4f;
}

double I_s(const vector<double>& _x) {
    int n = _x.size();
    if (n == 0) { return 0.; }
    double sum = f(_x[0]) + f(_x[n - 1]), h = _x[1] - _x[0];

    for (int i = 1; i < n - 1; ++i) {
        if (i % 2 == 0) {
            sum += (2 * f(_x[i]));
        }
        else {
            sum += (4 * f(_x[i]));
        }
    }
    return h / 3 * sum;
}

double abs_maximum(const vector<double>& _d4f, vector<double> _x) {
    int n = _d4f.size();
    double max = 0;
    int ind = 0;
    for (int i = 0; i < n; ++i) {
        if (abs(_d4f[i]) > abs(max)) {
            max = _d4f[i];
            ind = i;
        }
    }
    cout << max << " " << ind << "\n";
    cout << _x[ind - 2] << "\n";
    cout << f(_x[ind - 1]) << "\n";
    cout << _x[ind] << "\n";
    cout << f(_x[ind + 1]) << "\n";
    cout << f(_x[ind + 2]) << "\n";
    cout << (f(_x[ind - 2]) - 4*f(_x[ind - 1]) + 6*f(_x[ind]) - 4*f(_x[ind + 1]) + f(_x[ind + 2])) / pow(_x[1] - _x[0], 4);
    return max;
}

double R_s(const vector<double>& _x) {
    return -pow(_x[1] - _x[0], 4) * (_x[_x.size() - 1] - _x[0]) / 18 * abs_maximum(D4f(_x), _x);
}

// не дошли руки доделать
vector<double> searching_for_step(double _a, double _b, double _eps) {
    double h = 0.1;
    vector<double> result;


        int n = (_b - _a) / h + 1;
        vector<double> x(n);
        for (int i = 0; i < n; ++i) {
            x[i] = _a + i * h;
        }
        double R = R_s(x);
        /*if (abs(R) <= _eps) {
            result = x;
            x.clear();
        }*/
        h /= 2;
        x.clear();

    return x;
}

void Solution() {
    double eps = 1e-4;
    double a = 0, b = 1;

    
    while (abs(f(b)) > eps) { b += 1; }
    vector<double> x = searching_for_step(a, b, eps);

    cout << I_s(x);

}

int main() {
    Solution();

    return 0;
}