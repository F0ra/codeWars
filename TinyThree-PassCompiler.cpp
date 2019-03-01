#include <regex>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

using std::cout;
using std::string;
using std::vector;

struct AST;
struct Compiler;

struct AST {
    AST(string opType, AST* leftOperand, AST* rightOperand)
        :op(opType), a(leftOperand), b(rightOperand) {}
    AST(string opType, int val):op(opType), n(val) {
        a = nullptr;
        b = nullptr;
    }
    int n{};
    AST *a;
    AST *b;
    string op;

    friend Compiler;
    friend string jsonParse(AST *ast);
    friend string jsonParseForHTML(AST *ast);
};

struct Compiler {
    vector <string> compile(string program) {
        return pass3 (pass2 (pass1 (program)));
    }

    // Turn a program string into a vector of tokens.  Each token
    // is either '[', ']', '(', ')', '+', '-', '*', '/', a variable
    // name or a number (as a string)
    vector <string> tokenize(string program) {
        static std::regex re("[-+*/()[\\]]|[A-Za-z]+|\\d+");
        std::sregex_token_iterator it(program.begin(), program.end(), re);
        return vector <string> (it, std::sregex_token_iterator());
    }

    // Returns an un-optimized AST
    AST *pass1(string program);

    // Returns an AST with constant expressions reduced
    AST *pass2(AST *ast);

    // Returns assembly instructions
    vector <string> pass3(AST *ast);


 private:
    // pass1 methods --->
    vector<string> trimExp(const vector<string> exp);
    vector<string> createArgsList(const vector<string> &tokens);
    AST* createAST(vector<string> exp, const vector<string> args);
    vector<string> getExpressionFromProgram(const vector<string> &tokens);

    // pass2 methods --->
    void visitor(AST *&node);
    void resetN(AST *&node);
    
};
// Pass 1 ===================================================================>>>

AST *Compiler::pass1(string program) {
    vector <string> tokens = tokenize(program);
    vector<string> args = createArgsList(tokens);
    vector<string> exp =  getExpressionFromProgram(tokens);
    AST *root = createAST(exp, args);
    resetN(root);
    return root;
}

vector<string> Compiler::createArgsList(const vector<string> &tokens) {
    vector<string> args;
    bool stuckArgs = false;
    for (auto const token : tokens) {
        if (token == "]") return args;
        if (token == "[") {
            stuckArgs = true;
            continue;
        }
        if (stuckArgs) args.push_back(token);
    }
}

vector<string> Compiler::getExpressionFromProgram(
                                            const vector<string> &tokens) {
    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
        if (*it == "]") {
            vector<string> exp(++it, tokens.end());
            return exp;
        }
    }
}

vector<string> Compiler::trimExp(const vector<string> exp) {
    int parCount{};
    if (*(exp.begin()) == "(" && *(exp.rbegin()) == ")") {
        for (auto token = exp.begin(); token != exp.end(); ++token) {
            if (*token == "(") {
                parCount++;
            } else if (*token == ")") {
                parCount--;
            }

            if ((parCount == 0) && (token != exp.end()-1)) {
                break;
            } else if ((parCount == 0) && (token == exp.end()-1)) {
                vector<string> tExp(exp.begin() + 1, exp.end() - 1);
                return trimExp(tExp);
            }
        }
    }
    return exp;
}

AST* Compiler::createAST(const vector<string> exp, const vector<string> args) {
    vector<string> tExp = trimExp(exp);
    int parCount{};
    bool termOpChecked   = false;
    bool factorOpChecked = false;
    while (42) {
        for (auto token = tExp.rbegin(); token != tExp.rend(); ++token) {
            // check if exp is inside parentheses
            if        (*token == ")") {
                parCount++;
                continue;
            } else if (*token == "(") {
                parCount--;
                continue;
            } else if (parCount) {
                continue;
            }
            if (!termOpChecked && (*token == "*" || *token == "/")) continue;
            if  (*token == "+" || *token == "-" ||
                 *token == "*" || *token == "/") {
                vector<string> leftOperand(++token, tExp.rend());
                vector<string> rightOperand(tExp.rbegin(), --token);
                reverse(leftOperand.begin(), leftOperand.end());
                reverse(rightOperand.begin(), rightOperand.end());
                return new AST( (*token == "+") ? *token :
                                (*token == "-") ? *token :
                                (*token == "*") ? *token :
                                (*token == "/") ? *token :"error",
                                                createAST(leftOperand, args),
                                                createAST(rightOperand, args));
            }
            if (!factorOpChecked) continue;
            // check if factor is arg
            auto it = std::find(args.begin(), args.end(), *token);
            if (it != args.end()) {
                int index = std::distance(args.begin(), it);
                return new AST("arg", index);
            }
            // assume that token is a num
            return new AST("imm", std::stoi(*token));
        }
        if (!termOpChecked) {
            termOpChecked = true;
            continue;
        }
        if (!factorOpChecked) {
            factorOpChecked = true;
            continue;
        }
    }
}
//<<<===================================================================== Pass1

// Pass2 ====================================================================>>>
AST *Compiler::pass2(AST *ast) {
    // return ast;
    AST *optimizedAst = ast;
    visitor(optimizedAst);
    visitor(optimizedAst);
//     visitor(optimizedAst);
    resetN(optimizedAst);
    return optimizedAst;
}

void Compiler::resetN(AST *&node) {
    if (node->op != "imm" && node->op != "arg") {
        node->n = 0;
    }
    if (node->b) resetN(node->b);
    if (node->a) resetN(node->a);
    return;
}

void Compiler::visitor(AST *&node) {
    if (node->a && node->a->op == "imm" && node->b && node->b->op == "imm") {
        if (node->op == "+") {
            node->op = "imm";
            node->n = node->a->n + node->b->n;
            delete node->a;
            delete node->b;
            node->a = nullptr;
            node->b = nullptr;
        }
        if (node->op == "*") {
            node->op = "imm";
            node->n = node->a->n * node->b->n;
            delete node->a;
            delete node->b;
            node->a = nullptr;
            node->b = nullptr;
        }
        if (node->op == "-") {
            node->op = "imm";
            node->n = node->a->n - node->b->n;
            delete node->a;
            delete node->b;
            node->a = nullptr;
            node->b = nullptr;
        }
        if (node->op == "/") {
            node->op = "imm";
            node->n = node->a->n / node->b->n;
            delete node->a;
            delete node->b;
            node->a = nullptr;
            node->b = nullptr;
        }
    }
    if (node->b) visitor(node->b);
    if (node->a) visitor(node->a);
    return;
}
//<<<===================================================================== Pass2

// Pass3 ====================================================================>>>
vector <string> Compiler::pass3(AST *ast) {
    vector <string> assembly{};
    if          (ast->op == "arg") {
        assembly.push_back("AR " + std::to_string(ast->n));
        return assembly;
    } else if   (ast->op == "imm") {
        assembly.push_back("IM " + std::to_string(ast->n));
        return assembly;
    }

    vector <string> leftAssembly = pass3(ast->a);
    vector <string> rightAssembly = pass3(ast->b);
    bool isLeftPrimitive  = static_cast<bool>(leftAssembly.size()   == 1);
    bool isRightPrimitive = static_cast<bool>(rightAssembly.size()  == 1);

    if (isLeftPrimitive && isRightPrimitive) {
        assembly = rightAssembly;
        assembly.push_back("SW");
        assembly.push_back(*leftAssembly.begin());
    } else if (isLeftPrimitive && !isRightPrimitive) {
        assembly = rightAssembly;
        assembly.push_back("SW");
        assembly.push_back(*leftAssembly.begin());
    } else if (!isLeftPrimitive && isRightPrimitive) {
        assembly = leftAssembly;
        assembly.push_back("SW");
        assembly.push_back(*rightAssembly.begin());
        assembly.push_back("SW");
    } else if (!isLeftPrimitive && !isRightPrimitive) {
        assembly = leftAssembly;
        assembly.push_back("PU");
        assembly.reserve(assembly.size() + rightAssembly.size());
        assembly.insert(assembly.end(),
                        rightAssembly.begin(),
                        rightAssembly.end());
        assembly.push_back("SW");
        assembly.push_back("PO");
    }
    // add operator directive
    if        (ast->op == "+") {
        assembly.push_back("AD");
    } else if (ast->op == "-") {
        assembly.push_back("SU");
    } else if (ast->op == "*") {
        assembly.push_back("MU");
    } else if (ast->op == "/") {
        assembly.push_back("DI");
    }
    return assembly;
}
//<<<===================================================================== Pass3