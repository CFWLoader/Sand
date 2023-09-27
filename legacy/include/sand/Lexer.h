#pragma once
#include "SandGlobalDefines.h"
#include "token/Token.h"
#include <vector>
#include <iostream>
#include <regex>

namespace Sand {
    class Lexer {
    public:
        static const SandString REGEXP_PAT;
        Lexer(std::istream* Input);
        Token Read();
        Token Peek();
    protected:
        void ReadLine();
        void Tokenize();
    private:
        bool FillQueue(int Index);
        std::vector<Token> TokenFlow;
        bool HasMore;
        std::istream* Reader;
        std::regex Pattern;
    };
}