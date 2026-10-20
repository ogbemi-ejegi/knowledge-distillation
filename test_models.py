import torch

from models import TeacherNet, StudentNet

teacher = TeacherNet()
student = StudentNet()

dummy_input = torch.randn(64, 1, 28, 28)

teacher_output = teacher(dummy_input)
student_output = student(dummy_input)

print("Teacher Output Shape:", teacher_output.shape)
print("Student Output Shape:", student_output.shape)