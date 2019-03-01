#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <set>
#include <vector>
#include <iostream>

using std::set;
using std::vector;

class Sudoku;

struct Coordinates {
    size_t row;
    size_t col;
};

struct Cell {
    Cell(Coordinates coordinates):m_coordinates(coordinates) {
        for (size_t val{1}; val < 10; ++val) {
            m_possibleStates.insert(val);
        }
    }

    ~Cell(){ }

    void setRowNeighbors( vector<Cell*> rowNeighbors) {
        m_rowNeighbors = rowNeighbors;
    }

    void setColNeighbors( vector<Cell*> colNeighbors) {
        m_colNeighbors = colNeighbors;
    }

    void setSectorNeighbors( vector<Cell*> sectorNeighbors) {
        m_sectorNeighbors = sectorNeighbors;
    }

    void resetCell() {
        m_possibleStates.clear();
        for (size_t val{1}; val < 10; ++val) {
            m_possibleStates.insert(val);
        }
        m_val = 0;
        m_isDefined = false;
    }

    bool checkCellCollisions() {
        if (m_val == 0) return false;
        for (auto *cell : m_rowNeighbors) {
            if (cell->getVal() == m_val) return true;
        }
        for (auto *cell : m_colNeighbors) {
            if (cell->getVal() == m_val) return true;
        }
        for (auto *cell : m_sectorNeighbors) {
            if (cell->getVal() == m_val) return true;
        }
        return false;
    }


    int getVal() const { return m_val; }

    void setVal( int val) {
        m_val = val;
        m_isDefined = true;
        m_possibleStates.clear();
    }
    
    bool changeState() {
        m_stateChanged = false;
        if(m_isDefined) return m_stateChanged;
        
        auto isSubset = [](set<int> mainSet, set<int> setToCheck)->bool {
            if (mainSet.size() < setToCheck.size()) return false;
                for (auto it = setToCheck.begin(); it!=setToCheck.end(); ++it) {
                    if (mainSet.find(*it) == mainSet.end() ) return false;
                }
            return true;
        };

        auto delPosStates = [](set<int> posStates, vector <Cell*> preemptiveSet ,vector<Cell*> outerSet)->bool {
            bool isDelited = false;
            if( posStates.size() != preemptiveSet.size() || 
                                    posStates.size() == 1 || 
                                    outerSet.size() == 8 ) return false;
            for (auto *cell : outerSet) {
                for (auto it = posStates.begin(); it!=posStates.end(); ++it) {
                    isDelited = ( cell->m_possibleStates.erase(*it) || isDelited );
                }
            }
            return isDelited;
        };

        vector<Cell*> outerSet;
        vector<Cell*> preemptiveSet;

        for (auto *cell : m_rowNeighbors) {
            outerSet.clear();
            preemptiveSet.clear();
            if (cell->getVal()) {
                m_stateChanged = ( m_possibleStates.erase(cell->getVal()) || m_stateChanged );
            }
            if (cell->m_isDefined) continue;
            if (isSubset(m_possibleStates, cell->m_possibleStates)) {
                preemptiveSet.push_back(cell);
            } else {
                outerSet.push_back(cell);
            }
        }
        m_stateChanged = ( delPosStates(m_possibleStates, preemptiveSet ,outerSet) || m_stateChanged );

        for (auto *cell : m_colNeighbors) {
            outerSet.clear();
            preemptiveSet.clear();
            if (cell->getVal() ) {
                m_stateChanged = ( m_possibleStates.erase(cell->getVal()) || m_stateChanged );
            }
            if (cell->m_isDefined) continue;
            if (isSubset(m_possibleStates, cell->m_possibleStates)) {
                preemptiveSet.push_back(cell);
            } else {
                outerSet.push_back(cell);
            }
        }
        m_stateChanged = ( delPosStates(m_possibleStates, preemptiveSet, outerSet) || m_stateChanged );

        for (auto *cell : m_sectorNeighbors) {
            if (cell->getVal() ) {
                m_stateChanged = ( m_possibleStates.erase(cell->getVal()) || m_stateChanged );
            }
            if (cell->m_isDefined) continue;

            if (isSubset(m_possibleStates, cell->m_possibleStates)) {
                preemptiveSet.push_back(cell);
            } else {
                outerSet.push_back(cell);
            }
        }
        m_stateChanged = ( delPosStates(m_possibleStates, preemptiveSet, outerSet) || m_stateChanged );

        if (m_possibleStates.size() == 1) {
            set<int>::iterator it = m_possibleStates.begin();
            setVal(*it);
            m_possibleStates.clear();
            m_stateChanged = true;
        }

        auto posStateIsUniq = [] (size_t posState, vector<Cell*> neighbors)->bool {
            bool posStateUniq = true;
            for (auto *cell : neighbors) {
                if ( (cell->m_possibleStates.find(posState) != cell->m_possibleStates.end()) ) {
                    posStateUniq = false;
                    break;
                }
            }
            return posStateUniq;
        }; 

        for (auto it = m_possibleStates.begin(); it!=m_possibleStates.end(); ++it) {
            if( posStateIsUniq(*it, m_rowNeighbors) || 
                posStateIsUniq(*it, m_colNeighbors) || 
                posStateIsUniq(*it, m_sectorNeighbors) ) {
                
                setVal(*it);
                m_possibleStates.clear();
                m_stateChanged = true;
                return true;
            }
        }

        return m_stateChanged;
    }

 private:
    friend Sudoku;
    int m_val{0};
    bool m_isDefined{false};
    vector<Cell*> m_rowNeighbors{8};
    bool m_stateChanged{false};
    vector<Cell*> m_colNeighbors{8};
    vector<Cell*> m_sectorNeighbors{8};
    set<int> m_possibleStates;
    Coordinates m_coordinates;
};

class Sudoku {
 public:
    Sudoku() {
        initGrid();
    }

    ~Sudoku() {
        for (auto row : m_grid) {
            for (auto *cell : row) {
                if(cell!=nullptr) delete cell;
            }
        }
    }

    vector<vector<int>> solveSudoku(vector<vector<int>> grid) {
        resetGrid();
        setGrid(grid);
        if(checkCollisions()) return {};

        bool stateChanged;
        int count{0};
        do {
            stateChanged = false;
            ++count;
            for (auto row : m_grid) {
                for (auto *cell : row) {
                    stateChanged = (cell->changeState() || stateChanged);
                }
            }
        } while (stateChanged);

        return getGrid ();
    }

    vector<vector<int>> getGrid () {
        vector<vector<int>> grid{9};
        for (int row{0}; row < 9; ++row) {
            for (int col{0}; col < 9; ++col) {
                grid[row].push_back( m_grid[row][col]->getVal() );
            }
        }
        return grid;
    }


 private:

    vector<vector<Cell*>> m_grid{9};
    vector<vector<int>> m_prevGrid{};

    void resetGrid() {
        for (auto row : m_grid) {
            for (auto *cell : row) {
                if( cell != nullptr ) cell->resetCell();
            }
        }
    }

    void setGrid(vector<vector<int>> grid) {
        for (int row{0}; row < 9; ++row) {
            for (int col{0}; col < 9; ++col) {
                if (grid[row][col] == 0) continue;
                m_grid[row][col]->setVal(grid[row][col]);
            }
        }
    }

    bool checkCollisions() {
        for (auto row : m_grid) {
            for (auto *cell : row) {
                if (cell->checkCellCollisions()) {
                    return true;
                }
                    
            }
        }
        return false;
    }

    void initGrid() {
        for (int row{0}; row < 9; ++row) {
            for (int col{0}; col < 9; ++col) {
                m_grid[row].push_back(new Cell({row, col}));
            }
        }

        Cell *cell;
        for (int row{}; row < 9; ++row) {
            for (int col{}; col < 9; ++col) {
                cell = m_grid[row][col];
                setCellRowNeighbors(cell);
                setCellColNeighbors(cell);
                setCellSectorNeighbors(cell);
            }
        }

    }

    void setCellRowNeighbors(Cell *cell) {
        int cellRow = cell->m_coordinates.row;
        int cellCol = cell->m_coordinates.col;
        vector<Cell*> rowNeighbors;
        for (int col{0}; col < 9; ++col) {
            if(col == cellCol) continue;
            rowNeighbors.push_back(m_grid[cellRow][col]);
        }
        cell->setRowNeighbors(rowNeighbors);
    }

    void setCellColNeighbors(Cell *cell) {
        int cellRow = cell->m_coordinates.row;
        int cellCol = cell->m_coordinates.col;
        vector<Cell*> colNeighbors;
        for (int row{0}; row < 9; ++row) {
            if(row == cellRow) continue;
            colNeighbors.push_back(m_grid[row][cellCol]);
        }
        cell->setColNeighbors(colNeighbors);
    }

    void setCellSectorNeighbors(Cell *cell) {
        int cellRow = cell->m_coordinates.row;
        int cellCol = cell->m_coordinates.col;

        enum Sector {
            SECTOR1,
            SECTOR2,
            SECTOR3,
            SECTOR4,
            SECTOR5,
            SECTOR6,
            SECTOR7,
            SECTOR8,
            SECTOR9,
        };

        Sector sector;
        if (cellRow < 3 && cellCol < 3)                                 sector = SECTOR1;
        if (cellRow < 3 && cellCol > 2 && cellCol < 6)                  sector = SECTOR2;
        if (cellRow < 3 && cellCol > 5)                                 sector = SECTOR3;
        if (cellRow > 2 && cellRow < 6 && cellCol < 3)                  sector = SECTOR4;
        if (cellRow > 2 && cellRow < 6 && cellCol > 2 && cellCol < 6)   sector = SECTOR5;
        if (cellRow > 2 && cellRow < 6 && cellCol > 5)                  sector = SECTOR6;
        if (cellRow > 5 && cellCol < 3)                                 sector = SECTOR7;
        if (cellRow > 5 && cellCol > 2 && cellCol < 6)                  sector = SECTOR8;
        if (cellRow > 5 && cellCol > 5)                                 sector = SECTOR9;

        vector<std::pair<int,int>> sectorNeighborsCoordinate;
        
        switch (sector) {
            case SECTOR1:
                sectorNeighborsCoordinate = vector<std::pair<int,int>>({{0,0},{0,1},{0,2},
                                                                        {1,0},{1,1},{1,2},
                                                                        {2,0},{2,1},{2,2}});
                break;
            case SECTOR2:
                sectorNeighborsCoordinate = vector<std::pair<int,int>>({{0,3},{0,4},{0,5},
                                                                        {1,3},{1,4},{1,5},
                                                                        {2,3},{2,4},{2,5}});
                break;
            case SECTOR3:
                sectorNeighborsCoordinate = vector<std::pair<int,int>>({{0,6},{0,7},{0,8},
                                                                        {1,6},{1,7},{1,8},
                                                                        {2,6},{2,7},{2,8}});
                break;
            case SECTOR4:
                sectorNeighborsCoordinate = vector<std::pair<int,int>>({{3,0},{3,1},{3,2},
                                                                        {4,0},{4,1},{4,2},
                                                                        {5,0},{5,1},{5,2}});
                break;
            case SECTOR5:
                sectorNeighborsCoordinate = vector<std::pair<int,int>>({{3,3},{3,4},{3,5},
                                                                        {4,3},{4,4},{4,5},
                                                                        {5,3},{5,4},{5,5}});
                break;
            case SECTOR6:
                sectorNeighborsCoordinate = vector<std::pair<int,int>>({{3,6},{3,7},{3,8},
                                                                        {4,6},{4,7},{4,8},
                                                                        {5,6},{5,7},{5,8}});
                break;
            case SECTOR7:
                sectorNeighborsCoordinate = vector<std::pair<int,int>>({{6,0},{6,1},{6,2},
                                                                        {7,0},{7,1},{7,2},
                                                                        {8,0},{8,1},{8,2}});
                break;
            case SECTOR8:
                sectorNeighborsCoordinate = vector<std::pair<int,int>>({{6,3},{6,4},{6,5},
                                                                        {7,3},{7,4},{7,5},
                                                                        {8,3},{8,4},{8,5}});
                break;
            case SECTOR9:
                sectorNeighborsCoordinate = vector<std::pair<int,int>>({{6,6},{6,7},{6,8},
                                                                        {7,6},{7,7},{7,8},
                                                                        {8,6},{8,7},{8,8}});
                break;
        
    }
    
    vector<Cell*> sectorNeighbors;
    for (auto par : sectorNeighborsCoordinate) {
        if (par.first == cellRow && par.second == cellCol) continue;
        sectorNeighbors.push_back( m_grid[par.first][par.second] );
    }
    cell->setSectorNeighbors(sectorNeighbors);
    }
};

#endif // SOLVER_HPP
