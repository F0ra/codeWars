#include <set>
#include <vector>
#include <iostream>
#include <algorithm>

using std::set;
using std::cout;
using std::vector;

class Cell{
 public:
    struct Coordinate{
        int row;
        int col;
    };

    explicit Cell(int maxState, Coordinate coord)
    :m_coordinate(coord), m_maxState(maxState) {
        resetState();
    }
    int getState() const {return m_state;}
    set<int> getPossibleStates() {return m_possibleStates;}
    Coordinate getCellCoordinate() const {
        return m_coordinate;}
    void setState(int state) {
        m_state = state;
        if (m_state != 0) {
            m_possibleStates.clear();
            m_possibleStates.insert(state);
        }
    }
    void resetState() {
        m_possibleStates.clear();
        for (int i{1}; i <= m_maxState; ++i) {
            m_possibleStates.insert(i);
        }
        m_state = 0;
    }
    void setPossibleStates(const set<int> cellStateRestrictions) {
        for (auto it : cellStateRestrictions) {
            m_possibleStates.erase(it);
        }
        if (m_possibleStates.size() == 1) {
            m_state =  *m_possibleStates.begin();
        }
    }
    void resetPossibleStates(const set<int> possibleStates) {
        if (possibleStates.size() == 0) return;
        m_possibleStates = possibleStates;
        if (m_possibleStates.size() == 1) {
            m_state =  *m_possibleStates.begin();
        }
    }
    bool checkIfStateInSet(int state) const {
        return m_possibleStates.count(state);
    }

 private:
    int m_state{};
    int m_maxState{};
    set<int> m_possibleStates;
    Coordinate m_coordinate;
};

// =================== Skyscrapers =============================================
class Skyscrapers {
 public:
    Skyscrapers() {}
    ~Skyscrapers() {
        for (int row{}; row < m_gridSize; ++row) {
            for (int col{}; col < m_gridSize; ++col) {
                delete m_skyscrapersGrid[row][col];
            }
        }
    }
    void solve();
    void printSkyscrapers() const;
    vector<vector<int>> getSkyscrapersGrid() const;
    void setSkyscrapers(const vector<int> &clues);

 private:
    int m_gridSize{};
    vector<int> m_clues{};
    vector<int> m_topClues{};
    vector<int> m_leftClues{};
    vector<int> m_rightClues{};
    vector<int> m_bottomClues{};
    vector<int> m_kindsOfSkyscrambles{};
    vector<vector<Cell*>> m_skyscrapersGrid;
    bool isSolved();
    void resetGrid();
    void deleteGrid();
    void checkClues();
    void createGrid(int gridSize);
    void setStateRestriction(Cell *cell);
};

// Skyscrapers public methods ==============================================>>>>
void Skyscrapers::setSkyscrapers(const vector<int> &clues) {
    m_gridSize = clues.size()/4;
    if (m_clues.size() == clues.size()) {
        resetGrid();
    } else {
        createGrid(m_gridSize);
    }
    m_clues = clues;
    m_topClues.clear();
    m_topClues.reserve(m_gridSize);
    m_bottomClues.clear();
    m_bottomClues.reserve(m_gridSize);
    m_leftClues.clear();
    m_leftClues.reserve(m_gridSize);
    m_rightClues.clear();
    m_rightClues.reserve(m_gridSize);
    std::copy(m_clues.begin(), m_clues.begin() + m_gridSize,
                std::back_inserter(m_topClues));
    std::copy(m_clues.begin() + m_gridSize, m_clues.begin() + m_gridSize * 2,
                std::back_inserter(m_rightClues));
    std::copy(m_clues.rbegin() + m_gridSize, m_clues.rbegin() + m_gridSize * 2,
                std::back_inserter(m_bottomClues));
    std::copy(m_clues.rbegin(), m_clues.rbegin() + m_gridSize,
                std::back_inserter(m_leftClues));
}

void Skyscrapers::printSkyscrapers() const {
    cout << "\n============================\n";
    cout << "    ";
    for (auto it : m_topClues) cout << " " << it << " ";
    cout <<"\n    ";
    for (int i{}; i < m_gridSize; ++i) cout << "---";
    for (int i{}; i < m_gridSize; ++i) {
        cout << "\n "<< *(m_leftClues.begin() + i) << " |";
        for (int j{}; j < m_gridSize; ++j) {
            cout << " " << m_skyscrapersGrid[i][j]->getState() << " ";
        }
        cout << "| " << *(m_rightClues.begin() + i);
    }
    cout <<"\n    ";
    for (int i{}; i < m_gridSize; ++i) cout << "---";
    cout <<"\n    ";
    for (auto it : m_bottomClues) cout << " " << it << " ";
    cout <<"\n";
}

void Skyscrapers::solve() {
    int iter{};
    int iterLimit = 20;
    while ( !isSolved() && iter <= iterLimit ) {
        for (int i{}; i < m_gridSize; ++i) {
            for (int j{}; j < m_gridSize; ++j) {
                setStateRestriction(m_skyscrapersGrid[i][j]);
            }
        }
        checkClues();
        ++iter;
    }
}

vector<vector<int>> Skyscrapers::getSkyscrapersGrid() const {
    vector<vector<int>> grid;
    grid.resize(m_gridSize);
    for (int row{}; row < m_gridSize; ++row) {
        grid[row].resize(m_gridSize);
        for (int col{}; col < m_gridSize; ++col) {
            grid[row][col] = m_skyscrapersGrid[row][col]->getState();
        }
    }
    return grid;
}

// <<<=============================================== Skyscrapers public methods
// Skyscrapers private methods ==============================================>>>
void Skyscrapers::resetGrid() {
    for (int i{}; i < m_gridSize; ++i) {
        for (int j{}; j < m_gridSize; ++j) {
            m_skyscrapersGrid[i][j]->resetState();
        }
    }
}

void Skyscrapers::deleteGrid() {
    for (int i{}; i < m_skyscrapersGrid.size(); ++i) {
        for (int j{}; j < m_skyscrapersGrid[i].size(); ++j) {
            delete m_skyscrapersGrid[i][j];
        }
    }
    m_skyscrapersGrid.clear();
}

void Skyscrapers::createGrid(int gridSize) {
    deleteGrid();
    m_skyscrapersGrid.resize(gridSize);
    for (int i{}; i < gridSize; ++i) {
        for (int j{}; j < gridSize; ++j) {
            m_skyscrapersGrid[i].push_back(new Cell(gridSize, {i, j}));
        }
    }
    m_kindsOfSkyscrambles.clear();
    for (int kind{1}; kind <= gridSize; ++kind) {
        m_kindsOfSkyscrambles.push_back(kind);
    }
}

void Skyscrapers::setStateRestriction(Cell *cell) {
    set<int> cellStateRestrictions;
    set<int> cellStates;
    const int row = cell->getCellCoordinate().row;
    const int col = cell->getCellCoordinate().col;
    // check rows
    for (int i{}; i < m_gridSize; ++i) {
        if (i == col) continue;
        cellStateRestrictions.insert(m_skyscrapersGrid[row][i]->getState());
        for (auto it : m_skyscrapersGrid[row][i]->getPossibleStates()) {
            cellStates.insert(it);
        }
    }
    for (int state{1}; state <= m_gridSize; ++state) {
        if (!cellStates.count(state)) {
            cell->setState(state);
            return;
        }
    }
    cellStates.clear();
    // check colls
    for (int i{}; i < m_gridSize; ++i) {
        if (i == row) continue;
        cellStateRestrictions.insert(m_skyscrapersGrid[i][col]->getState());
        for (auto it : m_skyscrapersGrid[i][col]->getPossibleStates()) {
            cellStates.insert(it);
        }
    }
    for (int state{1}; state <= m_gridSize; ++state) {
        if (!cellStates.count(state)) {
            cell->setState(state);
            return;
        }
    }
    cell->setPossibleStates(cellStateRestrictions);
}

bool Skyscrapers::isSolved() {
    for (int row{}; row < m_gridSize; ++row) {
        for (int col{}; col < m_gridSize; ++col) {
            if (!m_skyscrapersGrid[row][col]->getState()) return false;
        }
    }
    return true;
}

void Skyscrapers::checkClues() {
    enum Glues{
        TOP_CLUES,
        RIGHT_CLUES,
        BOTTOM_CLUES,
        LEFT_CLUES,
        MAX_CLUES
    };
    vector<vector<Cell*>> tmpGlueOrientation;
    int glue;
    vector<int> currentClues;
    for (Glues side = TOP_CLUES; side != MAX_CLUES; side = Glues(side + 1)) {
        tmpGlueOrientation.clear();
        tmpGlueOrientation.resize(m_gridSize);
        switch (side) {
            case TOP_CLUES: {
                for (int row{}; row < m_gridSize; ++row) {
                    for (int col{}; col < m_gridSize; ++col) {
                        tmpGlueOrientation[row].push_back(
                                    m_skyscrapersGrid[col][row]);
                    }
                }
                currentClues = m_topClues;
                break;
            }
            case RIGHT_CLUES: {
                for (int row{}; row < m_gridSize; ++row) {
                    for (int col{}; col < m_gridSize; ++col) {
                        tmpGlueOrientation[row].push_back(
                                    m_skyscrapersGrid[row][m_gridSize-col-1]);
                    }
                }
                currentClues = m_rightClues;
                break;
            }
            case BOTTOM_CLUES: {
                for (int row{}; row < m_gridSize; ++row) {
                    for (int col{}; col < m_gridSize; ++col) {
                        tmpGlueOrientation[row].push_back(
                                    m_skyscrapersGrid[m_gridSize-col-1][row]);
                    }
                }
                currentClues = m_bottomClues;
                break;
            }
            case LEFT_CLUES: {
                tmpGlueOrientation = m_skyscrapersGrid;
                currentClues = m_leftClues;
                break;
            }
        }
        for (int row{}; row < m_gridSize; ++row) {
            glue = currentClues[row];
            if (!glue) continue;
            int tmpGlue{};
            int tmpMaxHeight{};
            vector<set<int>> possiblesStatesSets;
            possiblesStatesSets.resize(m_gridSize);
            bool skip;
            do {
                skip = false;
                tmpMaxHeight = tmpGlue = 0;
                for (int element{}; element < m_gridSize; ++element) {
                    if (!tmpGlueOrientation[row][element]->
                            checkIfStateInSet(m_kindsOfSkyscrambles[element])) {
                        skip = true;
                    }
                }
                if (skip) continue;
                for (int kind{}; kind < m_kindsOfSkyscrambles.size(); ++kind) {
                    if (m_kindsOfSkyscrambles[kind] > tmpMaxHeight) {
                        tmpMaxHeight = m_kindsOfSkyscrambles[kind];
                        tmpGlue++;
                    }
                }
                if (tmpGlue != glue) continue;
                for (int setEl{}; setEl < m_gridSize; ++setEl) {
                    possiblesStatesSets[setEl].insert(
                                                m_kindsOfSkyscrambles[setEl]);
                }
            } while (std::next_permutation(m_kindsOfSkyscrambles.begin(),
                                            m_kindsOfSkyscrambles.end()));
            for (int element{}; element < m_gridSize; ++element) {
                tmpGlueOrientation[row][element]->
                    resetPossibleStates(possiblesStatesSets[element]);
            }
        }
    }
}
// <<<============================================== Skyscrapers private methods

vector<vector<int>>  SolvePuzzle(const std::vector<int> &clues) {
    Skyscrapers sky;
    sky.setSkyscrapers(clues);
    sky.solve();
    return sky.getSkyscrapersGrid();
}

