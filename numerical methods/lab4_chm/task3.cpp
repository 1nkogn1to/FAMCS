#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#define INF 2147483647

using namespace std;

/*----------заполнение икса----------*/
void filling_x(vector<double>& _x, double _a, double _b, double _h) {
    int n = int((_b - _a) / _h) + 1;
    
    _x.clear(); _x.resize(n);

    for (int i = 0; i < n; ++i) {
        _x[i] = _a + i * _h;
    }
}
/*-----------------------------------*/

/*----------рассчёт основной части----------*/
double f(double _x) {
    return log(_x) / (_x * _x + 1);
}
double I_tr(vector<double> _x) {
    int n = _x.size();
    double sum = (f(_x[0]) + f(_x[n - 1])) / 2, h = _x[1] - _x[0];

    for (int i = 0; i < n; ++i) {
        sum += f(_x[i]);
    }
    return sum * h;
}
double I_s(vector<double> _x) {
    int n = _x.size();
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
/*------------------------------------------*/

/*----------рассчёт остатка----------*/ // вроде как не нужна эта часть кода, но можно оставить для проверки какой-нибудь
double ddf(double _x) {
    return ((-1 - 1 / (_x * _x) - 2 * log(_x)) * pow(_x * _x + 1, 2) - (_x + 1 / _x - 2 * _x * log(_x)) * 4 * _x * (_x*_x+1)) / pow(_x*_x+1, 4);
}
double ddddf(double _x) {
    return 2 * (- 3 - 14*pow(_x, 2) - 66*pow(_x, 6) - 77*pow(_x, 8) + 12*(pow(_x, 4) - 10*pow(_x, 6) + 5*pow(_x, 8)) * log(_x)) / (pow(_x, 4) * pow(1 + _x*_x, 5));
}
double maximum(vector<double> _x, double (*pFunc)(double)) {
    double maxF = -INF;

    int N = _x.size();
    for (int i = 0; i < N; ++i) {
        double current = (*pFunc)(_x[i]);
        if (current > maxF) maxF = current;
    }
    return maxF;
}
double R_tr(vector<double> _x) {
    double (*pFunc)(double) = nullptr, h = _x[1] - _x[0];
    pFunc = &ddf;
    return -pow(h, 2) * (_x[_x.size() - 1] - _x[0]) / 12 * maximum(_x, pFunc);
}
double R_s(vector<double> _x) {
    double (*pFunc)(double) = nullptr, h = _x[1] - _x[0];
    pFunc = &ddddf;
    return -pow(h, 4) * (_x[_x.size() - 1] - _x[0]) / 18 * maximum(_x, pFunc);
}
/*-----------------------------------*/

/*----------Правило Рунге----------*/
bool condition(double _I1, double _I2, double _h1, double _h2, double _eps) {
    static int m = 1;
    if (abs((_I2 - _I1) / (1 - pow(_h2 / _h1, m))) < _eps) {
        m += 2;
        return true;
    }
    return false;
}
/*---------------------------------*/

void main_loop(double _a, double _b, double _h1, double _h2, double _eps) {
    static int count = 1;
    vector<double> x1, x2;

    double (*pFunc)(vector<double>) = nullptr;

    count == 1 ? pFunc = &I_tr : pFunc = &I_s;

    while (true) {
        filling_x(x1, _a, _b, _h1);
        filling_x(x2, _a, _b, _h2);
        double I_h1 = pFunc(x1);
        double I_h2 = pFunc(x2);

        if (condition(I_h1, I_h2, _h1, _h2, _eps)) {
            cout << "Численное решение интеграла " << count++ << " методом с шагом " << _h1 << ": " << setprecision(10) << I_h1 << "\n";
            break;
        }
        _h1 /= 2;
        _h2 /= 2;
    }
}

void Solution() {
    double a = 1, b = 3, h1 = 0.1, h2 = h1 / 2, eps = 1e-8;

    main_loop(a, b, h1, h2, eps);
    main_loop(a, b, h1, h2, eps);
}

int main() {
    Solution();
    
    return 0;
}