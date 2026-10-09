"""Question metadata and conversation history survive benchmark measurement."""

from poolhouse.bench import concurrent, measure


def test_measure_preserves_expected_answers_and_model_label():
    def answer(question, client):
        return {'content': 'answer to ' + question, 'show': ['record:7']}
    rows = measure(answer, [{'q': 42, 'expect': [7]}, {'q': 'second', 'expect': ['record:7']}],
                   label='selected model', client=object(), trace=False)
    assert [(row.label, row.question, row.expected, row.conversation, row.turn) for row in rows] == [
        ('selected model', '42', ['7'], 0, 0),
        ('selected model', 'second', ['record:7'], 0, 0)]
    assert all(row.answer_chars == len('answer to ' + row.question) for row in rows)
    assert rows[1].shown == ['record:7']


def test_concurrent_rows_keep_positions_and_their_own_prior_answers():
    histories = {}
    def answer(question, client, *, turns=()):
        histories[question] = list(turns)
        return {'content': question + ' answer'}
    questions = [{'q': f'question-{index}', 'expect': [str(index)]} for index in range(4)]
    rows, _ = concurrent(answer, questions, conversations=2, turns=2,
                         label='shared model', client=object(), trace=False)
    assert [(row.label, row.conversation, row.turn, row.question, row.expected) for row in rows] == [
        ('shared model', 0, 0, 'question-0', ['0']), ('shared model', 0, 1, 'question-1', ['1']),
        ('shared model', 1, 0, 'question-2', ['2']), ('shared model', 1, 1, 'question-3', ['3'])]
    assert histories['question-0'] == histories['question-2'] == []
    assert histories['question-1'] == [{'role': 'user', 'content': 'question-0'},
                                       {'role': 'assistant', 'content': 'question-0 answer'}]
    assert histories['question-3'] == [{'role': 'user', 'content': 'question-2'},
                                       {'role': 'assistant', 'content': 'question-2 answer'}]
