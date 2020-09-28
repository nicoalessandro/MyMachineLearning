function [J grad] = nncf(Theta1, Theta2, k, X, y, lambda)

% Esta funcion tiene su origen en un ejercicio del curso de Machine Learning,
% el cual pedia como requisito una funcion que hiciera un FowardPropagation para definir 
% la funcion de costos y luego un Backpropagation para definir el gradiente.
% Aunque cambie los argumentos para mostrar solo la parte que tiene mi propio codigo.
% La funcion retorna el costo y el gradiente para ser utilizada como argumento en
% una funcion de optimizacion como puede ser fminunc().



% Setup some useful variables
m = size(X, 1); 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%Foward Propagation:
A1=X;%R(5000*400)
A1=[ones(m,1),A1]';%R(400*5000)
Z2=Theta1*A1;%R(25*401)*R(401*5000)=R(25*5000)
A2=sigmoid(Z2);%R(25*5000)
A2=[ones(1,m);A2];
Z3=Theta2*A2;%R(10*26)*R(26*5000)=R(10*5000)
A3=sigmoid(Z3);%R(10*5000)

% y={1,2,3,4,5,6,7,8,9,10} >>> y2={(1,0,...,0),(0,1,...,0),...,(0,0,...,1)}
y2=zeros(size(A3'));
for i=1:k
  y2(:,i)= (y==i);
endfor;

% Cost Function:
J= sum(log(A3(find(y2'==1)))(:))+sum(log(1-A3(find(y2'==0)))(:));
J=-J/m;
% Regularization:
reg=sum(sum( Theta1(:,2:size(Theta1,2)).^2 ))+sum(sum( Theta2(:,2:size(Theta2,2)).^2 ));
reg=lambda*reg/(2*m);
J=J+reg;

% Backpropagation:
%Theta1-->R(25*401)
%Theta2-->R(10*26)
deltha3=A3-y2';% R(10*5000)
deltha2=(Theta2')*deltha3.*A2.*(1-A2);% R(26*10)*R(10*5000).*R(26*5000)
Theta1_grad=deltha2(2:size(deltha2,1),:)*A1';%R(25*5000)*R(5000*401)= R(25*401)
Theta2_grad=deltha3*A2';%R(10*5000)*R(5000*26)*= R(10*26)

Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;
% +Regularization:
[f c]=size(Theta1);
Theta1Reg=lambda*[zeros(f,1),Theta1(:,2:c)]/m;
[f c]=size(Theta2);
Theta2Reg=lambda*[zeros(f,1),Theta2(:,2:c)]/m;

% Cambie los ciclos for originales del ejercicio por operaciones de matrices
% esto logro mejorar significativamente el tiempo de computo.

Theta1_grad=Theta1_grad+Theta1Reg;
Theta2_grad=Theta2_grad+Theta2Reg;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
