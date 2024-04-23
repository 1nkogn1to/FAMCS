#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

using namespace std;

complex<double> df(int size, complex<double> x, complex<double>* polinom) {
    complex<double>* result = new complex<double>[size - 1];
    for (int i = 0; i < size - 1; i++) { result[i] = polinom[i + 1]; }

    for (int i = 1; i < size - 1; i++) { result[i] *= i + 1; }

    complex<double> sum = complex<double>(0, 0);
    for (int i = 0; i < size - 1; i++) {
        sum += (result[i] * pow(x, i));
    }
    delete[] result;
    return sum;
}

complex<double> f(int size, complex<double> x, complex<double>* polinom) {
    complex<double> sum = complex<double>(0, 0);
    for (int i = 0; i < size; i++) {
        sum += (polinom[i] * pow(x, i));
    }
    return sum;
}

complex<double>* gorners_scheme(int size, complex<double> root, complex<double>* polinom) {
    complex<double>* temp = new complex<double>[size];
    temp[size - 1] = polinom[size - 1];
    for (int i = size - 2; i >= 0; i--) {
        temp[i] = root * temp[i + 1] + polinom[i];
    }    
    return temp;
}

complex<double>* newtons_method(int size, complex<double>* polinom, double accuracy) {
    complex<double>* roots = new complex<double>[size - 1];
    int _size = size;
    for (int i = 0; i < size - 3; i++) {
        complex<double> x0(-1);
        int counter = 0;
        for ( ; ; ) {
            complex<double> x1 = x0 - f(_size, x0, polinom) / df(_size, x0, polinom);
            ++counter;

            if (abs(x1 - x0) <= accuracy) {
                roots[i] = x1;
                break;
            }

            x0 = x1;
        }
        complex<double>* temp = gorners_scheme(_size, roots[i], polinom);
        
        --_size;
        delete[] polinom;
        polinom = new complex<double>[_size];

        for (int j = 0; j < _size; ++j) { polinom[j] = temp[j + 1]; }
        delete[] temp;
    }
    double D = polinom[1].real() * polinom[1].real() - 4 * polinom[0].real();

    roots[2] = (-polinom[1] - complex<double>(0, sqrt(-D))) / complex<double>(2, 0);
    roots[3] = (-polinom[1] + complex<double>(0, sqrt(-D))) / complex<double>(2, 0);

    return roots;
}

complex<double> abs_max(int size, complex<double>* roots) {
    complex<double> max(0, 0);
    for (int i = 0; i < size - 1; ++i) {
        if (abs(roots[i]) > abs(max)) {
            max = roots[i];
        }
    }
    return max;
}

int main() {
    freopen("input.txt", "r", stdin);
    int size;
    cin >> size;

    complex<double>* polinom = new complex<double>[size];
    for (int i = 0; i < size; ++i) {
        double temp;
        cin >> temp;
        polinom[i] = complex<double>(temp, 0);
    }

    complex<double>* roots = newtons_method(size, polinom, 1e-6);
    
    for (int i = 0; i < size - 1; ++i) {
        cout << roots[i] << "\n";
    }

    complex<double> maxim = abs_max(size, roots);

    cout << "Absolute maximum root: " << maxim << "\n";

    delete[] roots;
    delete[] polinom;
    return 0;
}