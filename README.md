# A_star_algorithm

<주요 Code 설명 및 활용법>

[7번 line의 'show_animation = True' 은 2D 이미지 형태로 작동하는 알고리즘을 보여줌 (True = open, False = close)]

[10번 line의 'use_jump_point = True'은 측정된 A*에서 추출된 각 vector 거리를 기반으로 distance 추출 가능 및 총 distance 추출 가능 (True = open, False = close and just show_animation A*)]


def main() 함수
[438, 439번 line의 start_x, start_y는 초기 지점의 position 값] (임의 설정 가능)
[443, 444번 line의 goal_x, goal_y는 목표 지점의 position 값] (임의 설정 가능)

[start position으로부터 goal point까지 지나간 [x,y] position]
line 490: print("location[x]: ", start_x, "->", rx)
line 491: print("location[y]: ", start_y, "->", ry)

[지나간 [x,y] position에 대한 vector]
print("vector[x,y]: ", vectors)

[추출된 총 distance vector]
print("The results of the distance: ", result_distance)

[A* 알고리즘의 걸린 시간(단위: sec)]
print("The results of the A* Algorithm's time: ", end-start)


{활용법}
Code Run 후에 나오는 2D 창이 끝나면 'q' 버튼을 누르면 됩니다.
ps. 코드 Run 후에 작동하는 시간이 조금 걸립니다.


*추가
사진을 보시게 되면 482~492 line까지 [x,y] position으로 각 지점에 위치한 wall(line 487)과 obstacle(line 492)의 지점을 확인하실수 있습니다.
