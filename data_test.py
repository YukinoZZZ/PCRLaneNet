from data_loader import Generator

loader = Generator()

index = 1
for raw_image, target_R2, target_R1, target_L2, target_L1, exist_label, test_image in loader.Generate():
    print('testing')
    print(index)
    index = index + 1