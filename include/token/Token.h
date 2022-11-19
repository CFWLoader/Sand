#pragma once

#include <iostream>
#include "StoneGlobalDefines.h"

namespace Stone{
	class IToken {
	public:
		virtual int getLineNumber() = 0;
		virtual bool isIdentifier() = 0;
		virtual bool isNumber() = 0;
		virtual bool isString() = 0;
		virtual int getNumber() = 0;
		virtual StoneString getText() = 0;
	};

	class Token: public IToken
	{
	public:
		static const Token ST_EOF;
		static const StoneString ST_EOL;
		virtual int getLineNumber();
		virtual bool isIdentifier();
		virtual bool isNumber();
		virtual bool isString();
		virtual int getNumber();
		virtual StoneString getText();
	protected:
		Token(uint32_t inLineNumber);
	private:
		uint32_t lineNumber;
	};
}

