#pragma once

#include <iostream>
#include "sand/SandGlobalDefines.h"

namespace Sand{
	class IToken {
	public:
		virtual int getLineNumber() const = 0;
		virtual bool isIdentifier() const = 0;
		virtual bool isNumber() const = 0;
		virtual bool isString() const = 0;
		virtual int getNumber() const = 0;
		virtual SandString getText() const = 0;
	};

	class Token: public IToken
	{
	public:
		static const Token ST_EOF;
		static const SandString ST_EOL;
		virtual int getLineNumber() const;
		virtual bool isIdentifier() const;
		virtual bool isNumber() const;
		virtual bool isString() const;
		virtual int getNumber() const;
		virtual SandString getText() const;
	protected:
		Token(uint32_t inLineNumber);
	private:
		uint32_t lineNumber;
	};
}

