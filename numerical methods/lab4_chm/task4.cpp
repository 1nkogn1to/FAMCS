#include <iostream>
#include <cmath>

using namespace std;

double ro(double x) {
    return sqrt(x) * exp(x);
}
double f(double x) {
    return sin(pow(x, 2)) / (1 + pow(log(x + 1), 2));
}
double ro_z(double t) {
    return sqrt(1 / t - 1) * exp(1 - 1 / t);
}
double f_z(double t) {
    return sin(pow(1 / t - 1, 2)) / (1 + pow(log(1 / t), 2)) * 1 / pow(t, 2);
}

// Для начала сократим область интегрирования поскольку подынтегральная функция стремится к нулю при стремлении к бесконечности
// Воспользуемся составной формулой 


void Solution() {
    double eps = 1e-4;
    double a = 0, b = 1;

    while (abs(f(b)) > eps) {
        b += 1;
    }
    cout << b << "\n";

    // Нашли b при котором функция достаточно мала
    // Теперь воспользуемся формулой расчёта остатка для составной квадратурной формулы,
    // поскольку она имеет наивысший порядок точности среди простейших формул
    // но для того, чтобы посчитать производную для данной функции необходимо много времени



}

int main() {
    Solution();

    return 0;
}