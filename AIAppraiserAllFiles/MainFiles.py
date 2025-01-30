from AiDirectory import AiRefactor

if __name__ == '__main__':
    Ai = AiRefactor.AI_Creait('/home/I/SchoolProject/AIAppraiserAllFiles/AiDirectory/AIAppraiser.keras')
    Ai.train_model_with_kfold()
    while True:
        print('Для того что бы остановить программу введите два раза 0')
        try:
            cup, hero = map(int, input('Введите сначала количество кубков аккаунта, а затем количество персонажей.\n').split(', '))
        except:
            print('Что-то пошло не так! Попробуйте еще раз!')
            continue
        prediction = Ai.predict(cup, hero)
        print(f'Предсказание для Cup: {cup}, Hero: {hero} - {prediction}')
        if cup == 0:
            break