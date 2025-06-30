/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/30/2025
 */
#include <iostream>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>

/**
 * Given integers y and p, find a number t such that (y * t) mod p = 1,
 * using the extended Euclidean algorithm.
 */
template <int p>
int getMultiplicativeInverse(const int y)
{
    int y_ = static_cast<int>(y.value);
    int t = 0; 
    int new_t = 1; 
    int r = p; 
    int new_r = y_; 
    while (new_r != 0)
    {
        int quotient = r / new_r; 
        int next_r = r - quotient * new_r; 
        int next_t = t - quotient * new_t;
        r = new_r; 
        new_r = next_r;  
        t = new_t;
        new_t = next_t;  
    }
    if (t < 0)
        t += p;

    return t; 
}

/**
 * Fields of characteristic p. 
 */
template <int p>
class Fp
{
    public:
        boost::multiprecision::gmp_rational value;

        Fp(const int x)
        {
            if (p == 0)
                this->value = static_cast<boost::multiprecision::gmp_rational>(x);
            else 
                this->value = static_cast<boost::multiprecision::gmp_rational>(x % p);  
        }

        ~Fp()
        {
        }

        Fp<p> operator+(const Fp<p>& x, const Fp<p>& y)
        {
            return Fp<p>(x.value + y.value); 
        }

        Fp<p>& operator+=(const Fp<p>& y)
        {
            this->value += y.value;
            this->value %= p;  
        }

        Fp<p> operator-(const Fp<p>& x, const Fp<p>& y)
        {
            return Fp<p>(x.value - y.value); 
        }

        Fp<p>& operator-=(const Fp<p>& y)
        {
            this->value -= y.value;
            this->value %= p;  
        }

        Fp<p> operator*(const Fp<p>& x, const Fp<p>& y)
        {
            return Fp<p>(x.value * y.value); 
        }

        Fp<p>& operator*=(const Fp<p>& y)
        {
            this->value *= y.value;
            this->value %= p;  
        }

        Fp<p> operator/(const Fp<p>& x, const Fp<p>& y)
        {
            // Check that y is nonzero 
            if (y == 0)
                throw std::runtime_error("Cannot divide by zero");

            // If p == 0, then simply return x / y
            if (p == 0)
                return Fp<p>(x.value / y.value);  

            // Otherwise, multiply x by the multiplicative inverse of y 
            int t = getMultiplicativeInverse<p>(y.value); 
            return Fp<p>(x_ * t);  
        }

        Fp<p>& operator/=(const Fp<p>& y)
        {
            // Check that y is nonzero 
            if (y == 0)
                throw std::runtime_error("Cannot divide by zero");

            // If p == 0, then simply return x / y
            if (p == 0)
                this->value /= y.value;

            // Otherwise, multiply x by the multiplicative inverse of y 
            int t = getMultiplicativeInverse<p>(y.value); 
            this->value *= t;
            this->value %= p; 
        } 
}

template <>
Fp<0>::Fp(const boost::multiprecision::gmp_rational x)
{
    this->value = x; 
}
