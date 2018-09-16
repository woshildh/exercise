#include<iostream>
#include"mycomplex.h"
using namespace std;

complex& calc(complex& x)
{
	return x;
}
int main()
{
	complex c1(2,0),c2(3,0);
	complex c3=calc(c1);

	cout << c1<< "," << c2 << "," << c3 << endl;
	
	cout<<c1+c2<<endl;
	cout<<c1-c2<<endl;
	cout<<c1*c2<<endl;
	cout<<c1/2<<endl;

	cout<<(c1+=c2)<<endl;
	cout<<conj(c1)<<endl;

	cout<<+c1<<endl;
	cout<<-c1<<endl;

	cout<<c1<<","<<c2<<c3<<","<<endl;
	system("pause");	
	return 0;
}


