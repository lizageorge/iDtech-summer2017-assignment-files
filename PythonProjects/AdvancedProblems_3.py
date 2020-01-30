speed = input("What speed were you going at? ")
birthday = input("What is your birthday? (mmdd, no spaces) ")
date_today = input("What is today's date? (mmdd, no spaces) ")

if int(birthday) == int(date_today):
    speed = int(speed) - 5

if int(speed) <= 60:
    print(0)
elif int(speed) > 60 and int(speed) <= 80:
    print(1)
else:
    print(2)

