#include <iostream>
#include "sand/token/Token.h"

using namespace std;
using namespace Sand;

int main()
{
	cout << Token::ST_EOF.getLineNumber() << endl;
	cout << "Hello Sand!" << endl;
}
