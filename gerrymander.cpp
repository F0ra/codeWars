
#include <vector>
#include <string>
#include <iostream>
#include <cassert>

using std::string;
using std::vector;

typedef vector<vector<int>> Grid;

class Test;

struct Coordinates {
    int row;
    int col;
};


struct GarryNode {
    GarryNode *parent = nullptr;
    GarryNode *root = nullptr;
    vector<GarryNode*> children;

    int length{};
    int district{};
    int votesCounter{};
    bool districtTail;
    Grid region;
    Coordinates coordinates;
};

class GarrySolver {
 public:
    GarrySolver():
        map(5,vector<int>(5)), 
        solvedRegion(5,vector<int>(5))
    {}

    string solve(const std::string &strMap) {
        map = strToGrid(strMap);
        createRootNode({0,0});
        if(solved) return gridToStr(solvedRegion);
        return "";
    }

    friend Test;

 private:
    unsigned int nodeCounter{};
    bool solved{false};
    Grid map;
    Grid solvedRegion;
    GarryNode *rootNode{nullptr};

    Grid strToGrid(const string mat) {
        Grid out;
        auto rowVec = [] (const string subStr) -> vector<int> {
            assert (subStr.size() == 5);
            vector<int> tmp(5);
            string x = "O";
            for (int i{}; i<5; ++i) {
                tmp[i] = (subStr[i] == x[0]) ? 1 : 0;
            }
            return tmp;
        };

        int cursor = 0;
        int prev = 0;
        while ( (cursor = mat.find("\n", prev)) != string::npos) {
            out.push_back(rowVec(mat.substr(prev, cursor-prev)));
            prev = cursor + 1;
        }
        if (prev <  mat.size()) out.push_back(rowVec(mat.substr(prev)));
        return out;
    }

    string gridToStr(const Grid region) {
        string out;
        for (int row{}; row<5; ++row) {
            if(row>0) out += "\n";
            for (int col{}; col<5; ++col) {
                out += std::to_string(region[row][col]);
            }
        }
        return out;
    }

    bool isCellValid (const Coordinates &c, const Grid &region) {
        if(c.row<0 || c.row>=5 || c.col<0 || c.col>=5) return false;
        return true;
    };

    bool isCellEmptyAndAdjacentToDistrict(const Coordinates &c, const Grid &region, int district) {
        if(!isCellValidAndEmpty(c, region)) return false;
        if( (isCellValid({c.row, c.col-1}, region) && region[c.row][c.col-1]==district ) ||
            (isCellValid({c.row-1, c.col}, region) && region[c.row-1][c.col]==district ) ||
            (isCellValid({c.row, c.col+1}, region) && region[c.row][c.col+1]==district ) ||
            (isCellValid({c.row+1, c.col}, region) && region[c.row+1][c.col]==district ) ) return true;
        return false;
    }

    bool isCellValidAndEmpty (const Coordinates &c, const Grid &region) {
        if(c.row<0 || c.row>=5 || c.col<0 || c.col>=5) return false;
        if(region[c.row][c.col]>0) return false;
        return true;
    };

    void fillAdjacentCell(Grid &region, Coordinates c, int *adjacentCellCounter) {
        if(!isCellValidAndEmpty({c.row, c.col}, region)) return;
        region[c.row][c.col] = 8;
        *adjacentCellCounter += 1;
        fillAdjacentCell(region, {c.row, c.col-1}, adjacentCellCounter);
        fillAdjacentCell(region, {c.row-1, c.col}, adjacentCellCounter);
        fillAdjacentCell(region, {c.row, c.col+1}, adjacentCellCounter);
        fillAdjacentCell(region, {c.row+1, c.col}, adjacentCellCounter);
    }

    bool isDeadEnd(Grid region) {
        int adjacentCellCounter = 0;
        for(int row{}; row<5; ++row) {
            for(int col{}; col<5; ++col) {
                if(region[row][col]==0) {
                    adjacentCellCounter = 0;
                    fillAdjacentCell(region, {row, col}, &adjacentCellCounter);
                    if(adjacentCellCounter%5 != 0) return true;
                }
            }
        }
        return false;
    }

    // bool isTwoInDistrict(GarryNode *tailNode) {
    //     int voters{};
    //     for(int row{}; row<5; ++row) {
    //         for(int col{}; col<5; ++col) {
    //             if(tailNode->region[row][col]==tailNode->district)
    //         }
    //     }        
    // }

    bool isSolved(Grid &map, Grid &region) {
        vector<int> districtsVec(5);

        for (int row{}; row<5; ++row) {
            for (int col{}; col<5; ++col) {
                if(map[row][col]) {
                    districtsVec[region[row][col]-1] += 1;
                }
            }
        }

        int majorDistricts{};
        for(auto val:districtsVec) (val>2) ? ++majorDistricts : true;
        return majorDistricts>2;
    }

    void createRootNode(const Coordinates c) {
        assert(c.row>=0 && c.row<5 && c.col>=0 && c.col<5);
        nodeCounter = 1;
        GarryNode *root = new GarryNode;
        
        root->coordinates.row = c.row;
        root->coordinates.col = c.col;

        root->root = root;
        root->length = 1;
        root->district = 1;
        root->districtTail = false;

        if(map[root->coordinates.row][root->coordinates.col]==1) {
            root->votesCounter = 1;
        } else {root->votesCounter = 0;}

        Grid tmp(5, vector<int>(5));
        root->region = tmp;
        root->region[c.row][c.col] = root->district;
        rootNode = root;
        createNodChildren(root);
    };

    GarryNode *createNode(GarryNode *parent,  Coordinates c) {
        assert(c.row>=0 && c.row<5 && c.col>=0 && c.col<5);
        ++nodeCounter;
        GarryNode *node = new GarryNode;
        if(solved) return node;
        node->parent = parent;
        node->coordinates.row = c.row;
        node->coordinates.col = c.col;
        node->length = parent->length + 1;
        node->district = (node->length-1)/5 + 1;
        node->districtTail = ((node->length)%5==0) ? true : false;
        node->region = parent->region;
        node->region[c.row][c.col] = node->district;
        if(node->parent->districtTail) {
            node->votesCounter = 0;
        } else {
            node->votesCounter = parent->votesCounter;
        }
        if(map[node->coordinates.row][node->coordinates.col]==1) node->votesCounter += 1;

        if(node->length==25) {
            if(isSolved(map, node->region)) {
                solved = true;
                solvedRegion = node->region;
            }
            return node;
        } else if(node->districtTail && node->votesCounter==2) {
            return node;
        } else if(node->districtTail && isDeadEnd(node->region)) {
            return node;
        } else if   ((node->districtTail||true) && 
                    (node->length==2 || node->length==4 || node->length==9) || node->length==14 || node->length==19) {
            createNodChildrenAdjacentToDistrict(node, node->district);
        } else {
            createNodChildren(node);
        }
        return node;
    }

    void createNodChildren(GarryNode *node) {
        if(isCellValidAndEmpty({node->coordinates.row, node->coordinates.col-1}, node->region)) 
            node->children.push_back(createNode(node, {node->coordinates.row, node->coordinates.col-1}));
        if(isCellValidAndEmpty({node->coordinates.row-1, node->coordinates.col}, node->region))
            node->children.push_back(createNode(node, {node->coordinates.row-1, node->coordinates.col}));
        if(isCellValidAndEmpty({node->coordinates.row, node->coordinates.col+1}, node->region))
            node->children.push_back(createNode(node, {node->coordinates.row, node->coordinates.col+1}));
        if(isCellValidAndEmpty({node->coordinates.row+1, node->coordinates.col}, node->region))
            node->children.push_back(createNode(node, {node->coordinates.row+1, node->coordinates.col}));
    }

    void createNodChildrenAdjacentToDistrict(GarryNode *node, int district) {
        for (int row{}; row < 5; ++row) {
            for (int col{}; col <5; ++col) {
                // std::cout<<"\nis adj: ("<<row<<", "<<col<<")"<<isCellEmptyAndAdjacentToDistrict({row, col}, node->region, district);
                if(isCellEmptyAndAdjacentToDistrict({row, col}, node->region, district))
                    node->children.push_back(createNode(node, {row, col}));
            }
        }
    }

};

std::string gerrymander(const std::string &map)
{
    GarrySolver solver;
    return solver.solve(map);
}