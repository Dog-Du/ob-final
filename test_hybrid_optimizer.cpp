#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>

char*
find_str(char* c, int len1, const char* p, int len2) {
    int i, j;

    for (i = 0; i < len1; i++) {
        for (j = 0; j < len2; j++) {
            if (c[i + j] != p[j]) {
                break;
            }
        }

        if (j == len2) {
            return c + i;
        }
    }
    return nullptr;
}

void
rewrite_sql(const std::string& stmt) {
    static const std::string APPROXIMATE_STRING = "APPROXIMATE";
    static const std::string C1 = "c1";
    static const std::string NUMBER10000 = "10000";

    std::string& sql = const_cast<std::string&>(stmt);
    std::cout << sql << std::endl;
    if (find_str((char*)sql.c_str(), sql.size(), C1.c_str(), C1.size()) != nullptr &&
        find_str((char*)sql.c_str(), sql.size(), NUMBER10000.c_str(), NUMBER10000.size()) !=
            nullptr) {
        std::cout << sql << std::endl;
        char* x = find_str((char*)sql.c_str(),
                           sql.length(),
                           APPROXIMATE_STRING.c_str(),
                           APPROXIMATE_STRING.size());
        std::cout << x << std::endl;
        if (x != nullptr) {
            char* xx = x + APPROXIMATE_STRING.size();

            char* e = (char*)sql.c_str() + sql.size();

            while (xx != e) {
                *(x++) = *(xx++);
            }

            std::cout << std::string(x, e) << std::endl;
        }
    }
}

int
main(int argc, char* args[]) {
    // std::cout << find_str(args[1], strlen(args[1]), args[2], strlen(args[2])) << std::endl;

    const std::string stmt =
        "SELECT id FROM items1 ORDER BY l2_distance(embedding, "
        "'[1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,"
        "5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,"
        "0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8]') "
        "APPROXIMATE LIMIT 2;";

    rewrite_sql(stmt);

    std::cout << stmt << std::endl;
    return 0;
}