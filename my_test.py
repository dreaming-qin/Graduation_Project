import argparse
parse = argparse.ArgumentParser()
parse.add_argument('--flag_int', type=int, default=2, help='flag_int')
# opt1 = parse.parse_args()
# print(opt1)


opt2 = parse.parse_known_args()
print(opt2)
print(opt2[0])
print(opt2[1])