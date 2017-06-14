from constants import NCAAF_TEAMS, NFL_TEAMS

def line_intersection(l1, l2):
    x1 = l1[0][0]
    y1 = l1[0][1]
    x2 = l1[1][0]
    y2 = l1[1][1]

    x3 = l2[0][0]
    y3 = l2[0][1]
    x4 = l2[1][0]
    y4 = l2[1][1]

    denom = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4)

    x = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/denom
    y = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/denom

    return x, y

def line_contains_null(line):
    for point in line:
        for coord in point:
            if coord is None:
                return True
    return False

def line_intersections(lines1, lines2):
    points = []
    for l1 in lines1:
        if line_contains_null(l1):
            continue
        for l2 in lines2:
            if line_contains_null(l2):
                continue
            x, y = line_intersection(l1, l2)
            points.append((x, y))

    return points

# The elegant solution
def filename_to_league(filename):
    if any(x in filename for x in NCAAF_TEAMS):
        return 'NCAAF'
    if any(x in filename for x in NFL_TEAMS):
        return 'NFL'

    return 'No league found'

