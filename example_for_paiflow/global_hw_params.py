BOARD_LIST = ["FLIP8", "BONDING003", "BONDING004", "BONDING008", "BONDING8"]

BOARD_NUMBER = "BONDING8"
FULL_CHANNEL_NUM = 4

if BOARD_NUMBER == "FLIP8":
    if FULL_CHANNEL_NUM == 16:   # not implemented
        oen_bin = "0" + "1" * 15 
    elif FULL_CHANNEL_NUM == 4:
        oen_bin = "0111"
elif BOARD_NUMBER == "BONDING003":
    oen_bin = "1000"
elif BOARD_NUMBER == "BONDING004":
    oen_bin = "1100"
elif BOARD_NUMBER == "BONDING008":
    oen_bin = "1110"
elif BOARD_NUMBER == "BONDING8":
    oen_bin = "1110"

for i in range(FULL_CHANNEL_NUM):
    if oen_bin[i] == "1":
        channel_mask_bin = "0" * i + "1" + "0" * (FULL_CHANNEL_NUM - i - 1)
        break

print("using board : {}".format(BOARD_NUMBER))
print("oen         : {}".format(oen_bin))
print("channel_mask: {}".format(channel_mask_bin))

oen = int(oen_bin, 2)
channel_mask = int(channel_mask_bin, 2)

assert oen > 0
assert ((channel_mask-1)&channel_mask) == 0 and (channel_mask > 0)

# bonding
# W6 W2 E6 E2

# FLIP CHIP
# E6 E2 W6 W2 

# 004 E2 bit error

# 004 W2 W6 E6