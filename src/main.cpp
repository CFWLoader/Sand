#include <iostream>
#include "sand/Lexer.h"

using namespace std;
using namespace Sand;

int main()
{
	// Lexer lexer1(MakeStr("dd"));
	Lexer lexer1;
	cout << lexer1.Peek().getLineNumber() << endl;
	cout << "Hello Sand!" << endl;
}
