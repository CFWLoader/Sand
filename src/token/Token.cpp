#include "token/Token.h"

using namespace Sand;

const Token Token::ST_EOF(-1);
const SandString Token::ST_EOL(MakeStr("\\n"));

int Token::getLineNumber() const
{
	return lineNumber;
}

bool Token::isIdentifier() const
{
	return false;
}

bool Token::isNumber() const
{
	return false;
}

bool Token::isString() const
{
	return false;
}

int Token::getNumber() const
{
	return 0;
}

SandString Token::getText() const
{
	return SandString();
}

Token::Token(uint32_t inLineNumber):
	lineNumber(inLineNumber)
{
}
