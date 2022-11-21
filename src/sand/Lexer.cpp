#include "sand/Lexer.h"

using namespace std;
using namespace Sand;

static const SandString REGEXP_PAT(MakeStr(""));

Lexer::Lexer(istream* Input):
     TokenFlow(),
     HasMore(false),
     Reader(Input),
     Pattern()
 {
    Tokenize();
 }

 Token Lexer::Read() {
    return Token::ST_EOF;
}

Token Lexer::Peek() {
    return Token::ST_EOF;
}

void Lexer::ReadLine() {
    
}

void Sand::Lexer::Tokenize()
{
    char linebuf[4096];
    while(!Reader->eof()) {
        Reader->getline(linebuf, 4096);
        std::cout << linebuf << endl;
    }
}

bool Lexer::FillQueue(int Index) {
    return false;
}
