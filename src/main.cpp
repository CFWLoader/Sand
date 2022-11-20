#include <iostream>
#include <fstream>
#include "sand/Lexer.h"

using namespace std;
using namespace Sand;

SandString MakeWinPath(const SandString& RawPath) {
	static const SandString WinProgramPrefex("..\\..\\..\\");
	return WinProgramPrefex + RawPath;
}

int main()
{
	ifstream inputFile;
	SandString ProgramPath(MakeStr("program1.sand"));
	inputFile.open(MakeWinPath(ProgramPath), ios::in);
	if (!inputFile.is_open()) {
		cerr << "Error opening program1.sand" << endl;
		return 0;
	}
	string lineOne;
	while (!inputFile.eof()) {
		inputFile >> lineOne;
		cout << lineOne << endl;
	}
	inputFile.close();
	return 0;
	// Lexer lexer1(MakeStr("dd"));
	// Lexer lexer1;
	// cout << lexer1.Peek().getLineNumber() << endl;
	// cout << "Hello Sand!" << endl;
}
