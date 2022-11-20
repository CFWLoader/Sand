#pragma once
#include "SandGlobalDefines.h"
#include "token/Token.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <regex>

namespace Sand {
    class Lexer {
    public:
        static const SandString REGEXP_PAT;
        // Lexer(std::ifstream& Input);
        Token Read();
        Token Peek();
    protected:
        void ReadLine();
    private:
        bool FillQueue(int Index);
        std::vector<Token> TokenFlow;
        bool HasMore;
        // std::istream Reader;
        std::regex Pattern;
    };
}