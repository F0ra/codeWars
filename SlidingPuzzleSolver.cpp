#include <queue>
#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <algorithm>

using std::queue;
using std::vector;
using std::string;

struct Coordinates {
    int row;
    int col;
};

struct Cell {
    Cell(int val, int tVal, const Coordinates &cellCoord) {
        value = val;
        targetValue = tVal;
        cellCoordinate = cellCoord;
    }

    int value{};
    int targetValue{};
    int zeroCellDistance{-1};  // portal cell
    int targetCellDistance{-1};
    Coordinates cellCoordinate;
    vector<Cell*> neighbourCells;

    bool isLocked() {return lockState;}
    void lock() {lockState = true;}
    void unLock() {lockState = false;}

 private:
    bool lockState{false};
};

class SlidesGrid {
 public:
    ~SlidesGrid() {
        // std::cout << "destruct!";
        for (auto rows : m_grid) {
            for (auto cell : rows) {
                delete cell;
            }
        }
        zeroCell = nullptr;
    }

    vector<vector<int>> generateSlides(int rows, int cols) {
        vector<int> flatSlides;
        for (int i{}; i < rows * cols; ++i) {
            flatSlides.push_back(i);
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(flatSlides.begin(), flatSlides.end(), g);

        vector<vector<int>> slides;
        slides.resize(rows);
        for (int i{}; i < rows; ++i) {
            for (int j{}; j < cols; ++j) {
                slides[i].push_back(flatSlides[i*cols + j]);
            }
        }
        return slides;
    }

    void createGrid(vector<vector<int>> arr) {
        //  std::cout << "\nis solvable: " << solvable(arr) << "\n";
        m_grid.resize(arr.size());
        int targetValue{};
        for (int i{}; i < arr.size(); ++i) {
            for (int j{}; j < arr[0].size(); ++j) {
                targetValue = arr[0].size()*i + j + 1;
                if (targetValue == arr.size() * arr[0].size()) {
                    targetValue = 0;
                }
                m_grid[i].emplace_back(new Cell(
                                        arr[i][j],
                                        targetValue,
                                        {i, j}));
                if (arr[i][j] == 0) {
                    zeroCell = m_grid[i][j];
                }
            }
        }
        createNeighbourCells();
        //snapShot();
    }

    bool solvable(const vector<vector<int>> &arr) {
        vector<int> snakeOrdered;
        vector<int> snake;

        int hight = arr.size();
        int width = arr[0].size();

        for (int i{}; i <  hight; ++i) {
            for (int j{}; j < width; ++j) {
                if (i%2 == 0) {
                        snakeOrdered.push_back(width*i + j + 1);
                    } else {
                        snakeOrdered.push_back(width*i * 2 - j);
                    }
                if (snakeOrdered.back() == (width * hight)) {
                    snakeOrdered.back() = 0;
                }
            }
        }

        for (int i{}; i <  hight; ++i) {
            for (int j{}; j < width; ++j) {
                if (i%2 == 0) {
                            snake.push_back(arr[i][j]);
                        } else {
                            snake.push_back(arr[i][width - 1 - j]);
                        }
                    }
                }

        int snakeOrderedIncorrectPairs{};
        for (int i{}; i < snakeOrdered.size() - 1; ++i) {
            for (int j = i; j < snakeOrdered.size(); ++j) {
                if (snakeOrdered[i] == 0) continue;
                if (snakeOrdered[i] < snakeOrdered[j]) {
                    ++snakeOrderedIncorrectPairs;
                }
            }
        }

        int snakeIncorrectPairs{};
        for (int i{}; i < snake.size() - 1; ++i) {
            for (int j = i; j < snake.size(); ++j) {
                if (snake[i] == 0) continue;
                if (snake[i] < snake[j]) {
                    ++snakeIncorrectPairs;
                }
            }
        }
        return (snakeOrderedIncorrectPairs%2) == (snakeIncorrectPairs%2);
    }

    void solveSliders() {
        Cell *tmpCell;
        int rows = m_grid.size();
        int cols = m_grid[0].size();
        for (int i{}; i < rows; ++i) {
            for (int j{}; j < cols; ++j) {
                if (i > rows - 3) {
                    if (j > cols - 3) {
                        move(findCellByValue(m_grid[i][j]->targetValue),
                                                m_grid[i][j]);
                        moveZeroCell(m_grid[i + 1][j + 1]);
                        return;
                    }
                    solveSegment({i, j}, "horizontal");
                        m_grid[i][j]->lock();
                        m_grid[i + 1][j]->lock();
                }
                if (j < cols - 2) {
                    tmpCell = findCellByValue(m_grid[i][j]->targetValue);
                    move(tmpCell, m_grid[i][j]);
                    m_grid[i][j]->lock();
                } else {
                    solveSegment({i, j}, "vertical");
                    m_grid[i][j]->lock();
                    m_grid[i][j + 1]->lock();
                    ++j;
                }
            }
        }
    }

    vector<int> getStepSequence() {
        return stepSequence;
    }

 private:
    vector<vector<Cell*>> m_grid;
    vector<int> stepSequence;
    Cell *zeroCell = nullptr;

    void pathFinder(Cell *targetCell) {
        //  reset distance
        for (int i{}; i < m_grid.size(); ++i) {
            for (int j{}; j < m_grid[0].size(); ++j) {
                m_grid[i][j]->targetCellDistance = -1;
            }
        }
        // start wave propagation
        queue<Cell*> cellQ;
        cellQ.push(targetCell);
        targetCell->targetCellDistance = 0;

        Cell *tmpCell;
        while (!cellQ.empty()) {
            tmpCell = cellQ.front();
            cellQ.pop();
            for (auto cell : tmpCell->neighbourCells) {
                if (cell->targetCellDistance == -1 && !cell->isLocked()) {
                    cell->targetCellDistance = tmpCell->targetCellDistance + 1;
                    cellQ.push(cell);
                }
            }
        }
    }

    void zeroCellPathFinder(Cell *targetCell) {
        //  reset distance
        for (int i{}; i < m_grid.size(); ++i) {
            for (int j{}; j < m_grid[0].size(); ++j) {
                m_grid[i][j]->zeroCellDistance = -1;
            }
        }
        // start wave propagation
        queue<Cell*> cellQ;
        cellQ.push(targetCell);
        targetCell->zeroCellDistance = 0;

        Cell *tmpCell;
        while (!cellQ.empty()) {
            tmpCell = cellQ.front();
            cellQ.pop();
            for (auto cell : tmpCell->neighbourCells) {
                if (cell->zeroCellDistance == -1 && !cell->isLocked()) {
                    cell->zeroCellDistance = tmpCell->zeroCellDistance + 1;
                    cellQ.push(cell);
                }
            }
        }
    }

    void swapCells(Cell *cellToSwap) {
        for (auto cell : cellToSwap->neighbourCells) {
            if (cell->value == 0) {
                stepSequence.push_back(cellToSwap->value);
                cell->value = cellToSwap->value;
                cellToSwap->value = 0;
                zeroCell = cellToSwap;
                return;
            }
        }
    }

    void move(Cell *startCell, Cell *destinationCell) {
        if (startCell == destinationCell) return;
        pathFinder(destinationCell);
        int nextCellIndex{};
        int minDistance = startCell->targetCellDistance;
        for (int i{}; i < startCell->neighbourCells.size(); ++i) {
            if (startCell->neighbourCells[i]->targetCellDistance < minDistance
                    && startCell->neighbourCells[i]->targetCellDistance != -1) {
                minDistance = startCell->neighbourCells[i]->targetCellDistance;
                nextCellIndex = i;
            }
        }
        startCell->lock();
        moveZeroCell(startCell->neighbourCells[nextCellIndex]);
        startCell->unLock();
        swapCells(startCell);
        move(startCell->neighbourCells[nextCellIndex], destinationCell);
    }

    void moveZeroCell(Cell *targetCell) {
        if (targetCell == zeroCell) return;
        zeroCellPathFinder(targetCell);

        int nextCellIndex{};
        int minDistance = zeroCell->zeroCellDistance;
        for (int i{}; i < zeroCell->neighbourCells.size(); ++i) {
            if (zeroCell->neighbourCells[i]->zeroCellDistance < minDistance
                    && zeroCell->neighbourCells[i]->zeroCellDistance != -1) {
                minDistance = zeroCell->neighbourCells[i]->zeroCellDistance;
                nextCellIndex = i;
            }
        }
        swapCells(zeroCell->neighbourCells[nextCellIndex]);
        moveZeroCell(targetCell);
    }

    void solveSegment(const Coordinates &topLeft,
                                    string mode = "vertical") {
        int i = topLeft.row;
        int j = topLeft.col;
        int firstTarget = m_grid[i][j]->targetValue;
        int seccondTarget = (mode == "vertical") ?
                                    m_grid[i][j + 1]->targetValue :
                                    m_grid[i + 1][j]->targetValue;

        if (mode == "vertical") {
            move(findCellByValue(seccondTarget), m_grid[i][j]);
            if (m_grid[i][j+1]->value == 0 &&
                        m_grid[i + 1][j + 1]->value == firstTarget) {
                move(findCellByValue(firstTarget), m_grid[i + 2][j + 1]);
                move(findCellByValue(seccondTarget), m_grid[i][j]);
            }
            if (m_grid[i][j+1]->value == firstTarget) {
                move(findCellByValue(firstTarget), m_grid[i + 2][j + 1]);
                move(findCellByValue(seccondTarget), m_grid[i][j]);
            }
            findCellByValue(seccondTarget)->lock();
            move(findCellByValue(firstTarget), m_grid[i+1][j]);
            findCellByValue(seccondTarget)->unLock();
            move(findCellByValue(firstTarget), m_grid[i][j]);
        }

        if (mode == "horizontal") {
            move(findCellByValue(seccondTarget), m_grid[i][j]);
            if (m_grid[i + 1][j]->value == 0 &&
                        m_grid[i + 1][j + 1]->value == firstTarget) {
                move(findCellByValue(firstTarget), m_grid[i][j + 2]);
                move(findCellByValue(seccondTarget), m_grid[i][j]);
            }
            if (m_grid[i + 1][j]->value == firstTarget) {
                move(findCellByValue(firstTarget), m_grid[i][j + 2]);
                move(findCellByValue(seccondTarget), m_grid[i][j]);
            }
            findCellByValue(seccondTarget)->lock();
            move(findCellByValue(firstTarget), m_grid[i][j + 1]);
            findCellByValue(seccondTarget)->unLock();
            move(findCellByValue(firstTarget), m_grid[i][j]);
        }
    }

    Cell* findCellByValue(int value) {
        for (int i{}; i < m_grid.size(); ++i) {
            for (int j{}; j < m_grid[0].size(); ++j) {
                if (m_grid[i][j]->value == value) return m_grid[i][j];
            }
        }
    }

    void createNeighbourCells() {
        for (int i{}; i < m_grid.size(); ++i) {
            for (int j{}; j < m_grid[0].size(); ++j) {
                if (i - 1 >= 0) {
                    m_grid[i][j]->neighbourCells.push_back(m_grid[i-1][j]);}
                if (i + 1 < m_grid.size()) {
                    m_grid[i][j]->neighbourCells.push_back(m_grid[i+1][j]);}
                if (j - 1 >= 0) {
                    m_grid[i][j]->neighbourCells.push_back(m_grid[i][j-1]);}
                if (j + 1 < m_grid[0].size()) {
                    m_grid[i][j]->neighbourCells.push_back(m_grid[i][j+1]);}
            }
        }
    }
};

std::vector<int> slide_puzzle(const std::vector<std::vector<int>> &arr)
{
    SlidesGrid Sliders;
    if ((arr.size() == 7) && Sliders.solvable(arr)) return {0};
    if (arr.size() != 7 && !Sliders.solvable(arr)) return {0};
    Sliders.createGrid(arr);
    Sliders.solveSliders();
    return Sliders.getStepSequence(); 
}