#include "Token.h"

using namespace Stone;

const Token Token::ST_EOF(-1);
const StoneString Token::ST_EOL(MakeStr("\\n"));

int Stone::Token::getLineNumber()
{
	return lineNumber;
}

bool Stone::Token::isIdentifier()
{
	return false;
}

bool Stone::Token::isNumber()
{
	return false;
}

bool Stone::Token::isString()
{
	return false;
}

int Stone::Token::getNumber()
{
	return 0;
}

StoneString Stone::Token::getText()
{
	return StoneString();
}

Stone::Token::Token(uint32_t inLineNumber):
	lineNumber(inLineNumber)
{
}
