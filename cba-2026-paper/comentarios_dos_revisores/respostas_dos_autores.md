# Respostas dos Autores

**SUBMISSION:** 2227
**TITLE:** Error-Based Active Disturbance Rejection Control for Parkinson's Disease Tremor Suppression in Wrist Considering Upper Limb Dynamics

## Comentários dos Revisores

### Revisor 1

1. aprofundar a discussão sobre implementação em tempo real e limitações computacionais do EADRC;
2. incluir uma análise estatística mais detalhada dos resultados;
3. discutir possíveis limitações do modelo linearizado utilizado; e 
4. melhorar a escrita em alguns trechos da seção de resultados, que em certos momentos fica mais descritiva do que analítica.

### Revisor 2

5. Authors should update the reference list.
6. The introduction is too long. Authors should focus on the disease's control aspects.
7. Please explain or provide a reference for the values of a1, a2, and a3 presented after (2).
8. How realistic is assuming the involuntary torque is purely sinusoidal? How will the presence of other harmonics impact the method?
9. Please define w after (11).
10. In the sentence before (22), the correct is "presented in Figure 2."
11. The discussion of the results is too brief. Reducing the number of methods in the comparison may provide the necessary space to enhance the discussion.

## Respostas aos comentários

- [x] 1. uma breve discussão sobre a implementação em tempo real e limitações computacionais do EADRC foi incluída na fundamentação teórica.
- [x] 2. o p-valor da significância estatística da diferença de performance entre os métodos é apresentado em nova tabela na seção dos resultados.
- [x] 3. uma breve discussão do modelo linearizado utilizado foi incluída na fundamentação teórica, após a eq. (1).
- [x] 4. mais análises foram incluídas na seção de resultados e discussões.
- [x] 5. algumas referências foram removidas, novas referências foram incluídas, erros de digitação e formatação nas referências foram corrigidos.
- [x] 6. a introdução foi refatorada, tornando-a mais breve e focada nas metodologias de controle do tremor de Parkinson.
- [x] 7. referência às fórmulas dos centroides foi incluída no parágrafo que segue a equação (2).
- [x] 8. comentários a respeito da modelagem do tremor foram feitos na fundamentação teórica, após (7).
- [x] 9. "w" é um erro de digitação e foi removido da equação.
- [x] 10. a frase foi corrigida.
- [x] 11. as análises dos resultados foram aprofundadas, e a metodologia de controle de pior performance foi removida.

## Observações

- As edições relativas ao comentário 11 têm sinergia com as dos itens 2, 4, 5 e 6.