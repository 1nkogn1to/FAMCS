#include <iostream>
#include <cmath>

using namespace std;


double f(int _degree, double _x) {
    return pow(_x, _degree);
}

double df(int _degree, double _x) {
    return _degree * pow(_x, _degree - 1);
}

double If(double _a, double _b, int _degree) {
    return pow(_b, _degree + 1) / (_degree + 1) - pow(_a, _degree + 1) / (_degree + 1);
}

double quadrature_formula(int _degree) {
    return 1./30 * (7 * (f(_degree, 0.) + f(_degree, 1.)) + 16 * f(_degree, 1./2) - 1./2 * (df(_degree, 1.) - df(_degree, 0.)));
}

void Solution() {
    int p = 1;
    double a = 0., b = 1.;

    while (If(a, b, p) == quadrature_formula(p)) {
        p++;
    }

    cout << "Квадратурная формула имеет алгебраический порядок точности равный " << p << "\n";
}

int main() {
    Solution();

    return 0;
}