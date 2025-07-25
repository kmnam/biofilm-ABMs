/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/22/2025
 */
#ifndef FIELDS_OF_ARBITRARY_CHARACTERISTIC_HPP
#define FIELDS_OF_ARBITRARY_CHARACTERISTIC_HPP

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>

typedef boost::multiprecision::mpq_rational Rational; 

using std::abs; 
using boost::multiprecision::abs;

/**
 * A type-agnostic modulo operator that returns a value between 0 and p - 1
 * (inclusive). 
 */
template <typename T, int p>
int mod(const T x)
{
    return (x < 0 ? static_cast<int>(-x) % p : static_cast<int>(x) % p);
}

/**
 * A lightweight implementation of the Z/2Z field. 
 */
class Z2
{
    public:
        bool value; 

        Z2()
        {
            this->value = 0; 
        }

        Z2(const bool x)
        {
            this->value = x; 
        }

        Z2(const int x)
        {
            this->value = mod<int, 2>(x); 
        }

        Z2(const Z2& x)
        {
            this->value = x.value; 
        }

        ~Z2()
        {
        }

        // ---------------------------------------------------------- //
        //                    ASSIGNMENT OPERATORS                    //
        // ---------------------------------------------------------- //
        Z2& operator=(const bool& x)
        {
            this->value = x;
            return *this;  
        }

        Z2& operator=(const int& x)
        {
            this->value = mod<int, 2>(x); 
            return *this;  
        }
        
        Z2& operator=(const Z2& x)
        {
            this->value = x.value;
            return *this;  
        }

        // ---------------------------------------------------------- //
        //              EQUALITY AND INEQUALITY OPERATORS             //
        // ---------------------------------------------------------- //
        bool operator==(const bool& x) const 
        {
            return (this->value == x); 
        }
        
        bool operator==(const int& x) const 
        {
            return (this->value == mod<int, 2>(x)); 
        }

        bool operator==(const Z2& x) const
        {
            return (this->value == x.value); 
        }

        bool operator!=(const bool& x) const 
        {
            return !(*this == x); 
        }

        bool operator!=(const int& x) const 
        {
            return !(*this == x); 
        }

        bool operator!=(const Z2& x) const 
        {
            return !(*this == x); 
        }

        // ---------------------------------------------------------- //
        //                    CONVERSION OPERATORS                    //
        // ---------------------------------------------------------- //
        explicit operator bool() const 
        {
            return this->value; 
        }

        explicit operator int() const 
        {
            return static_cast<int>(this->value); 
        }

        explicit operator double() const 
        {
            return static_cast<double>(this->value); 
        }

        explicit operator Rational() const 
        {
            return static_cast<Rational>(this->value); 
        }

        // ---------------------------------------------------------- //
        //                 GREATER-THAN AND LESS-THAN                 // 
        // ---------------------------------------------------------- //
        bool operator>(const Z2& x) const 
        {
            return this->value > x.value; 
        }

        bool operator>=(const Z2& x) const 
        {
            return this->value >= x.value; 
        }

        bool operator<(const Z2& x) const 
        {
            return this->value < x.value; 
        }

        bool operator<=(const Z2& x) const 
        {
            return this->value <= x.value; 
        } 

        // ---------------------------------------------------------- //
        //                  INCREMENT AND DECREMENT                   //
        // ---------------------------------------------------------- //
        Z2& operator++()
        {
            this->value = (!this->value ? 1 : 0);
            return *this; 
        }
        
        Z2& operator--()
        {
            this->value = (!this->value ? 1 : 0);
            return *this;  
        }

        // ---------------------------------------------------------- //
        //                           NEGATION                         //
        // ---------------------------------------------------------- //
        Z2 operator-() const 
        {
            return Z2(this->value); 
        }

        // ---------------------------------------------------------- //
        //                    ARITHMETIC OPERATORS                    //
        // ---------------------------------------------------------- //
        Z2& operator+=(const bool& y)
        {
            this->value = (this->value ^ y);
            return *this;  
        }

        Z2& operator+=(const int& y)
        {
            this->value = (this->value ^ mod<int, 2>(y)); 
            return *this; 
        }

        Z2& operator+=(const Z2& y)
        {
            this->value = (this->value ^ y.value); 
            return *this;  
        }
        
        Z2& operator-=(const bool& y)
        {
            this->value = (this->value ^ y);
            return *this;  
        }

        Z2& operator-=(const int& y)
        {
            this->value = (this->value ^ mod<int, 2>(y)); 
            return *this; 
        }

        Z2& operator-=(const Z2& y)
        {
            this->value = (this->value ^ y.value); 
            return *this;  
        }

        Z2& operator*=(const bool& y)
        {
            this->value = (this->value && y); 
            return *this; 
        }

        Z2& operator*=(const int& y)
        {
            this->value = (this->value && mod<int, 2>(y)); 
            return *this; 
        }

        Z2& operator*=(const Z2& y)
        {
            this->value = (this->value && y.value); 
            return *this; 
        }

        Z2& operator/=(const bool& y)
        {
            if (!y)
                throw std::runtime_error("Cannot divide by zero");
            else
                return *this;
        } 
        
        Z2& operator/=(const int& y)
        {
            if (mod<int, 2>(y) == 0)
                throw std::runtime_error("Cannot divide by zero");
            else
                return *this;
        } 

        Z2& operator/=(const Z2& y)
        {
            if (!y.value)
                throw std::runtime_error("Cannot divide by zero");
            else
                return *this;
        } 
}; 

Z2 operator+(const Z2& x, const bool& y)
{
    return Z2(x.value ^ y); 
}

Z2 operator+(const Z2& x, const int& y)
{
    return Z2(x.value ^ mod<int, 2>(y)); 
}

Z2 operator+(const bool& x, const Z2& y)
{
    return y + x; 
}

Z2 operator+(const int& x, const Z2& y)
{
    return y + x; 
}

Z2 operator+(const Z2& x, const Z2& y)
{
    return Z2(x.value ^ y.value); 
}

Z2 operator-(const Z2& x, const bool& y)
{
    return x + y; 
}

Z2 operator-(const Z2& x, const int& y)
{
    return x + y; 
}

Z2 operator-(const bool& x, const Z2& y)
{
    return x + y; 
}

Z2 operator-(const int& x, const Z2& y)
{
    return x + y; 
}

Z2 operator-(const Z2& x, const Z2& y)
{
    return x + y; 
}

Z2 operator*(const Z2& x, const bool& y)
{
    return Z2(x.value && y); 
}

Z2 operator*(const Z2& x, const int& y)
{
    return Z2(x.value && mod<int, 2>(y)); 
}

Z2 operator*(const bool& x, const Z2& y)
{
    return y * x; 
}

Z2 operator*(const int& x, const Z2& y)
{
    return y * x; 
}

Z2 operator*(const Z2& x, const Z2& y)
{
    return Z2(x.value && y.value); 
}

Z2 operator/(const bool& x, const Z2& y)
{
    if (!y.value)
        throw std::runtime_error("Cannot divide by zero");
    else 
        return Z2(x); 
}

Z2 operator/(const int& x, const Z2& y)
{
    if (!y.value)
        throw std::runtime_error("Cannot divide by zero");
    else 
        return Z2(x); 
}

Z2 operator/(const Z2& x, const bool& y)
{
    if (!y)
        throw std::runtime_error("Cannot divide by zero");
    else 
        return Z2(x);  
}

Z2 operator/(const Z2& x, const int& y)
{
    if (mod<int, 2>(y) == 0)
        throw std::runtime_error("Cannot divide by zero");
    else 
        return Z2(x);  
}

Z2 operator/(const Z2& x, const Z2& y)
{
    if (!y.value)
        throw std::runtime_error("Cannot divide by zero");
    else 
        return Z2(x);  
}

bool operator==(const bool& x, const Z2& y)
{
    return (x == y.value); 
}

bool operator==(const int& x, const Z2& y)
{
    return (mod<int, 2>(x) == y.value); 
}

bool operator!=(const bool& x, const Z2& y)
{
    return (x != y.value); 
}

bool operator!=(const int& x, const Z2& y)
{
    return (mod<int, 2>(x) != y.value); 
}

bool operator^(const Z2& x, const Z2& y)
{
    return x.value ^ y.value; 
}

std::ostream& operator<<(std::ostream& out, const Z2& x)
{
    return out << x.value; 
}

Z2 abs(const Z2& x)
{
    return Z2(x); 
}

/**
 * Given integers y and p, find a number t such that (y * t) mod p = 1,
 * using the extended Euclidean algorithm.
 */
template <int p>
int getMultiplicativeInverse(const int y)
{
    if (y == 0)
        throw std::runtime_error("Multiplicative inverse does not exist for zero"); 

    int t = 0; 
    int new_t = 1; 
    int r = p; 
    int new_r = y; 
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
    private:
        void reset()
        {
            if (p != 0 && (this->value < 0 || this->value >= p))
                this->value = Rational(mod<Rational, p>(this->value));
        } 

    public:
        Rational value; 

        Fp()
        {
            this->value = 0; 
        }

        Fp(const int x)
        {
            if (p == 0)
                this->value = Rational(x); 
            else 
                this->value = Rational(mod<int, p>(x)); 
        }

        Fp(const Rational x)
        {
            if (p == 0)
                this->value = x; 
            else if (boost::multiprecision::denominator(x) == 1)
                this->value = Rational(mod<Rational, p>(x)); 
            else    // x is not an integer and p is nonzero 
                throw std::runtime_error(
                    "Cannot assign non-integer value in field of finite "
                    "characteristic"
                );  
        }

        Fp(const Fp<p>& x)
        {
            this->value = x.value; 
        }

        ~Fp()
        {
        }

        // ---------------------------------------------------------- //
        //                    ASSIGNMENT OPERATORS                    //
        // ---------------------------------------------------------- //
        Fp<p>& operator=(const int& x)
        {
            if (p == 0)
                this->value = x;
            else
                this->value = mod<int, p>(x); 
            return *this;  
        }
        
        Fp<p>& operator=(const Fp<p>& x)
        {
            this->value = x.value;
            return *this;  
        }

        // ---------------------------------------------------------- //
        //              EQUALITY AND INEQUALITY OPERATORS             //
        // ---------------------------------------------------------- //
        bool operator==(const int& x) const 
        {
            if (p == 0)
                return (this->value == x); 
            else
                return (mod<Rational, p>(this->value - x) == 0); 
        }

        bool operator==(const Rational& x) const 
        {
            if (p == 0)
            {
                return (this->value == x);
            } 
            else
            {
                if (boost::multiprecision::denominator(x) != 1)
                    return false;
                else  
                    return (mod<Rational, p>(this->value - x) == 0); 
            } 
        }

        bool operator==(const Fp<p>& x) const
        {
            if (p == 0)
                return (this->value == x.value);
            else 
                return (mod<Rational, p>(this->value - x.value) == 0); 
        }

        bool operator!=(const int& x) const 
        {
            return !(*this == x); 
        }

        bool operator!=(const Rational& x) const 
        {
            return !(*this == x); 
        }

        bool operator!=(const Fp<p>& x) const 
        {
            return !(*this == x); 
        }

        // ---------------------------------------------------------- //
        //                 GREATER-THAN AND LESS-THAN                 // 
        // ---------------------------------------------------------- //
        bool operator>(const Fp<p>& x) const 
        {
            return this->value > x.value; 
        }

        bool operator>=(const Fp<p>& x) const 
        {
            return this->value >= x.value; 
        }

        bool operator<(const Fp<p>& x) const 
        {
            return this->value < x.value; 
        }

        bool operator<=(const Fp<p>& x) const 
        {
            return this->value <= x.value; 
        } 

        // ---------------------------------------------------------- //
        //                  INCREMENT AND DECREMENT                   //
        // ---------------------------------------------------------- //
        Fp<p>& operator++()
        {
            this->value++; 
            if (p != 0 && this->value == p)
                this->value = 0; 
            return *this; 
        }
        
        Fp<p>& operator--()
        {
            this->value--;
            if (p != 0 && this->value == -1)
                this->value = p - 1;
            return *this;  
        }

        // ---------------------------------------------------------- //
        //                           NEGATION                         //
        // ---------------------------------------------------------- //
        Fp<p> operator-() const 
        {
            return Fp<p>(-this->value); 
        }

        // ---------------------------------------------------------- //
        //                    ARITHMETIC OPERATORS                    //
        // ---------------------------------------------------------- //
        Fp<p>& operator+=(const int& y)
        {
            this->value += y; 
            return *this; 
        }

        Fp<p>& operator+=(const Rational& y)
        {
            if (p != 0 && boost::multiprecision::denominator(y) != 1)
                throw std::runtime_error(
                    "Cannot add with non-integer number in field of finite "
                    "characteristic"
                ); 
            this->value += y;
            this->reset();  
            return *this; 
        }

        Fp<p>& operator+=(const Fp<p>& y)
        {
            this->value += y.value;
            this->reset(); 
            return *this;  
        }

        Fp<p>& operator-=(const int& y)
        {
            this->value -= y;
            this->reset();  
            return *this; 
        }

        Fp<p>& operator-=(const Rational& y)
        {
            if (p != 0 && boost::multiprecision::denominator(y) != 1)
                throw std::runtime_error(
                    "Cannot subtract with non-integer number in field of "
                    "finite characteristic"
                ); 
            this->value -= y;
            this->reset();  
            return *this; 
        }

        Fp<p>& operator-=(const Fp<p>& y)
        {
            this->value -= y.value;
            this->reset(); 
            return *this; 
        }

        Fp<p>& operator*=(const int& y)
        {
            this->value *= y;
            this->reset();  
            return *this; 
        }

        Fp<p>& operator*=(const Rational& y)
        {
            if (p != 0 && boost::multiprecision::denominator(y) != 1)
                throw std::runtime_error(
                    "Cannot multiply with non-integer number in field of "
                    "finite characteristic"
                ); 
            this->value *= y;
            this->reset();  
            return *this; 
        }

        Fp<p>& operator*=(const Fp<p>& y)
        {
            this->value *= y.value;
            this->reset();
            return *this;  
        }

        Fp<p>& operator/=(const int& y)
        {
            // If p == 0, then simply return x / y
            if (p == 0)
            {
                // Check that y is nonzero 
                if (y == 0)
                    throw std::runtime_error("Cannot divide by zero");
                this->value /= y;
            }
            else    // Otherwise, multiply x by the multiplicative inverse of y
            {
                // Check that y falls within the range 0, ..., p - 1 and is nonzero
                int y_ = ((y < 0 || y >= p) ? mod<int, p>(y) : y);
                if (y_ == 0)
                    throw std::runtime_error("Cannot divide by zero"); 
                int t = getMultiplicativeInverse<p>(y_); 
                this->value *= t;
                this->reset(); 
            }
            return *this;  
        }

        Fp<p>& operator/=(const Rational& y)
        {
            // If p == 0, then simply return x / y
            if (p == 0)
            {
                // Check that y is nonzero
                if (y == 0)
                    throw std::runtime_error("Cannot divide by zero");
                this->value /= y;
            }
            else    // Otherwise, multiply x by the multiplicative inverse of y
            {
                // Check that y is an integer and falls within the range 1, ..., p - 1
                if (boost::multiprecision::denominator(y) != 1)
                    throw std::runtime_error(
                        "Cannot divide with non-integer number in field of "
                        "finite characteristic"
                    ); 
                int y_ = ((y < 0 || y >= p) ? mod<Rational, p>(y) : static_cast<int>(y));
                if (y_ == 0)
                    throw std::runtime_error("Cannot divide by zero"); 
                int t = getMultiplicativeInverse<p>(y_); 
                this->value *= t;
                this->reset(); 
            }
            return *this;  
        }

        Fp<p>& operator/=(const Fp<p>& y)
        {
            // Check that y is nonzero 
            if (y == 0)
                throw std::runtime_error("Cannot divide by zero");

            // If p == 0, then simply return x / y
            if (p == 0)
            {
                this->value /= y.value;
            }
            else    // Otherwise, multiply x by the multiplicative inverse of y
            {
                int t = getMultiplicativeInverse<p>(static_cast<int>(y.value)); 
                this->value *= t;
                this->reset(); 
            }
            return *this;  
        } 
};

template <int p>
Fp<p> operator+(const Fp<p>& x, const int& y)
{
    return Fp<p>(x.value + y); 
}

template <int p>
Fp<p> operator+(const Fp<p>& x, const Rational& y)
{
    return Fp<p>(x.value + y); 
}

template <int p>
Fp<p> operator+(const int& x, const Fp<p>& y)
{
    return y + x; 
}

template <int p>
Fp<p> operator+(const Rational& x, const Fp<p>& y)
{
    return y + x; 
}

template <int p>
Fp<p> operator+(const Fp<p>& x, const Fp<p>& y)
{
    return Fp<p>(x.value + y.value); 
}

template <int p>
Fp<p> operator-(const Fp<p>& x, const int& y)
{
    return Fp<p>(x.value - y); 
}

template <int p>
Fp<p> operator-(const Fp<p>& x, const Rational& y)
{
    return Fp<p>(x.value - y); 
}

template <int p>
Fp<p> operator-(const int& x, const Fp<p>& y)
{
    return y - x; 
}

template <int p>
Fp<p> operator-(const Rational& x, const Fp<p>& y)
{
    return y - x; 
}

template <int p>
Fp<p> operator-(const Fp<p>& x, const Fp<p>& y)
{
    return Fp<p>(x.value - y.value); 
}

template <int p>
Fp<p> operator*(const Fp<p>& x, const int& y)
{
    return Fp<p>(x.value * y); 
}

template <int p>
Fp<p> operator*(const int& x, const Fp<p>& y)
{
    return y * x; 
}

template <int p>
Fp<p> operator*(const Rational& x, const Fp<p>& y)
{
    return y * x; 
}

template <int p>
Fp<p> operator*(const Fp<p>& x, const Rational& y)
{
    return Fp<p>(x.value * y); 
}

template <int p>
Fp<p> operator*(const Fp<p>& x, const Fp<p>& y)
{
    return Fp<p>(x.value * y.value); 
}

template <int p>
Fp<p> operator/(const int& x, const Fp<p>& y)
{
    // Check that y is nonzero 
    if (y == 0)
        throw std::runtime_error("Cannot divide by zero");

    // If p == 0, then simply return x / y
    if (p == 0)
    {
        return Fp<p>(x / y.value); 
    }
    else    // Otherwise, multiply x by the multiplicative inverse of y
    {
        // Check that x falls within the range 0, ..., p - 1
        int x_ = ((x < 0 || x >= p) ? mod<int, p>(x) : x); 
        int t = getMultiplicativeInverse<p>(static_cast<int>(y.value)); 
        return Fp<p>(x_ * t);
    } 
}

template <int p>
Fp<p> operator/(const Fp<p>& x, const int& y)
{
    // If p == 0, then simply return x / y
    if (p == 0)
    {
        // Check that y is nonzero 
        if (y == 0)
            throw std::runtime_error("Cannot divide by zero");
        return Fp<p>(x.value / y);
    }
    else    // Otherwise, multiply x by the multiplicative inverse of y
    {
        // Check that y falls within the range 0, ..., p - 1 and is nonzero
        int y_ = ((y < 0 || y >= p) ? mod<int, p>(y) : y);
        if (y_ == 0)
            throw std::runtime_error("Cannot divide by zero"); 
        int t = getMultiplicativeInverse<p>(y_); 
        return Fp<p>(x.value * t);
    } 
}

template <int p>
Fp<p> operator/(const Rational& x, const Fp<p>& y)
{
    // Check that y is nonzero 
    if (y == 0)
        throw std::runtime_error("Cannot divide by zero");
    
    // If p == 0, then simply return x / y
    if (p == 0)
    {
        return Fp<p>(x / y.value); 
    }
    else    // Otherwise, multiply x by the multiplicative inverse of y
    {
        // Check that x is an integer and falls within the range 0, ..., p - 1
        if (boost::multiprecision::denominator(x) != 1)
            throw std::runtime_error(
                "Cannot divide with non-integer number in field of finite "
                "characteristic"
            );
        int x_ = ((x < 0 || x >= p) ? mod<Rational, p>(x) : static_cast<int>(x)); 
        int t = getMultiplicativeInverse<p>(static_cast<int>(y.value)); 
        return Fp<p>(x_ * t);
    } 
}

template <int p>
Fp<p> operator/(const Fp<p>& x, const Rational& y)
{
    // If p == 0, then simply return x / y
    if (p == 0)
    {
        // Check that y is nonzero 
        if (y == 0)
            throw std::runtime_error("Cannot divide by zero");
        return Fp<p>(x.value / y);
    }
    else    // Otherwise, multiply x by the multiplicative inverse of y
    {
        // Check that y is an integer and falls within the range 1, ..., p - 1
        if (boost::multiprecision::denominator(y) != 1)
            throw std::runtime_error(
                "Cannot divide with non-integer number in field of finite "
                "characteristic"
            );
        int y_ = ((y < 0 || y >= p) ? mod<Rational, p>(y) : static_cast<int>(y));
        if (y_ == 0)
            throw std::runtime_error("Cannot divide by zero"); 
        int t = getMultiplicativeInverse<p>(y_); 
        return Fp<p>(x.value * t);
    } 
}

template <int p>
Fp<p> operator/(const Fp<p>& x, const Fp<p>& y)
{
    // Check that y is nonzero 
    if (y == 0)
        throw std::runtime_error("Cannot divide by zero");

    // If p == 0, then simply return x / y
    if (p == 0)
    {
        return Fp<p>(x.value / y.value);
    } 
    else    // Otherwise, multiply x by the multiplicative inverse of y
    {
        int t = getMultiplicativeInverse<p>(static_cast<int>(y.value));
        return Fp<p>(x.value * t);
    } 
}

template <int p>
std::ostream& operator<<(std::ostream& out, const Fp<p>& x)
{
    return out << x.value; 
}

template <int p>
Fp<p> abs(const Fp<p>& x)
{
    return Fp<p>(abs(x.value)); 
}

namespace Eigen {

template <>
struct NumTraits<Z2> : NumTraits<bool>
{
    typedef Z2 Real;
    typedef Z2 NonInteger; 
    typedef Z2 Nested; 

    enum
    {
        IsComplex = 0, 
        IsInteger = 1, 
        IsSigned = 0,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    }; 
}; 

template <>
struct NumTraits<Fp<0> > : NumTraits<Rational>
{
    typedef Fp<0> Real; 
    typedef Fp<0> NonInteger; 
    typedef Fp<0> Nested; 

    enum
    {
        IsComplex = 0, 
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 5,    // TODO
        AddCost = 10,
        MulCost = 20
    };
};

template <>
struct NumTraits<Fp<2> > : NumTraits<int>
{
    typedef Fp<2> Real; 
    typedef Fp<2> NonInteger; 
    typedef Fp<2> Nested; 

    enum
    {
        IsComplex = 0, 
        IsInteger = 1,
        IsSigned = 0,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

}   // namespace Eigen 

#endif
