#include "sand/Lexer.h"

using namespace std;
using namespace Sand;

static const SandString REGEXP_PAT(MakeStr(""));

// Lexer::Lexer(ifstream& Input) :
//     TokenFlow(),
//     HasMore(false),
//     Reader(Input),
//     Pattern()
// {
// }

Token Lexer::Read() {
    return Token::ST_EOF;
}

Token Lexer::Peek() {
    return Token::ST_EOF;
}

void Lexer::ReadLine() {
    
}

bool Lexer::FillQueue(int Index) {

}
