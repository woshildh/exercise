#ifndef __MYCOMPLEX__
#define __MYCOMPLEX__

class complex;
complex& 
  __doapl(complex* ths,const complex& r);
complex&
  __doami(complex* ths,const complex& r);
complex&
  __doaml(complex* ths,const complex& r);

class complex
{
	public:
		complex(double r=0,double i=0):re(r),im(i) { }
		complex& operator +=(const complex&);
		complex& operator -=(const complex&);
		complex& operator *=(const complex&);	
		complex& operator /=(const complex&);
		double real() const {return re;}
		double imag() const {return im;}
	private:
		double re,im;
	friend complex& __doapl(complex *,const complex&);
	friend complex& __doaml(complex *,const complex&);
	friend complex& __doami(complex *,const complex&);
};

inline complex&
__doapl(complex* ths,const complex& r)
{
	ths->re += r.re;
	ths->im +=r.im;
	return *ths;
}
inline complex&
complex::operator +=(const complex& r)
{
	return __doapl(this,r);
}

inline complex&
__doami(complex *ths,const complex& r)
{
	ths->re+=r.re;
	ths->im+=r.im;
	return *ths;
}
inline complex&
complex::operator -=(const complex& r)
{
	return __doami(this,r);
}

inline complex&
__doaml(complex *ths,const complex& r)
{
	double f=ths->re * r.re - ths->im * r.im;

	ths->im=ths->re * r.im + ths->im * r.re;
	ths->re=f;
}
inline complex&
complex::operator *=(const complex& r)
{
	return __doaml(this,r);
}

inline double
imag(const complex& r)
{
	return r.imag();
}
inline double
real(const complex& r)
{
	return r.real();
}

inline complex
operator +(const complex& r,const complex& i)
{
	return complex(r.real()+i.real(),r.imag()+i.imag());
}
inline complex
operator +(const complex& r,double d)
{
	return complex(r.real()+d,r.imag());
}
inline complex
operator +(double d,const complex& r)
{
	return complex(d+r.real(),r.imag());
}

inline complex
operator -(const complex& x,const complex& y)
{
	return complex(x.real()-y.real(),x.imag()-y.imag());
}
inline complex
operator -(const complex& x,double d)
{
	return complex(x.real()-d,x.imag());
}
inline complex
operator -(double d,const complex& x)
{
	return complex(d-x.real(),x.imag());
}

inline complex
operator *(const complex& x,double d)
{
	return complex(d*x.real(),d*x.imag());
}
inline complex
operator *(double d,const complex& x)
{
	return complex(d*x.real(),d*x.imag());
}
inline complex
operator *(complex& x,complex& y)
{
	return complex(x.real()*y.real()-x.imag()*y.imag(),
		x.real()*y.imag()+x.imag()*y.real());
}

inline complex
operator /(complex& x,double d)
{
	return complex(x.real()/d,x.imag()/d);
}

inline complex
operator +(const complex& x)
{
	return x;
}
inline complex
operator -(const complex& x)
{
	return complex(-x.real(),-x.imag());
}

inline bool
operator ==(const complex& x,const complex& y)
{
	return (x.real()==y.real()) && (x.imag()==y.imag());
}
inline bool
operator ==(const complex& x,double d)
{
	return (x.real()==d) && (x.imag()==0);
}
inline bool
operator ==(double d,const complex& x)
{
	return (x.real()==d) && (x.imag()==0);
}

inline bool
operator !=(const complex& x,const complex& y)
{
	return (x.real()!=y.real()) || (x.imag() != y.real());
}
inline bool
operator !=(const complex& x,double d)
{
	return (x.real()!=d) || (x.imag()!=0);
}
inline bool
operator !=(double d,const complex& x)
{
	return (x.real()!=d) || (x.imag()!=0);
}

#include <iostream>
using namespace std;
inline ostream&
operator <<(ostream& os,const complex& x)
{
	return os<<"("<<x.real()<<","<<x.imag()<<")";
}


inline complex
conj(const complex& x)
{
	return complex(x.real(),-x.imag());
}

#endif



