#include "ParserUtil.h"

#include <cctype>
#include "ConfigUsing.h"

void ignoreComment(string& line) {
	size_t pos = line.find_first_of("#");
	if (pos != string::npos) {
		line = line.substr(0, pos);
	}
}

void ignorePrefixWhitespace(string& line) {
	size_t i = 0;
	while (std::isspace(line[i])) {
		++i;
	}
	line = line.substr(i);
}

Vec3f parseVertexCoord(string line) {
	std::vector<float> v;
	std::istringstream iss(line);
	std::copy(std::istream_iterator<float>(iss),
		std::istream_iterator<float>(),
		std::back_inserter(v));
	return Vec3f(v.data());
}

int parseInt(string& line) {
	string::size_type size;
	int i = stoi(line, &size);
	line = line.substr(size);
	return i;
}





