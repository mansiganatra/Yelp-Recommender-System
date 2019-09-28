from pyspark import SparkContext, StorageLevel
import sys
import json
import csv
import itertools
from time import time
import math
import random
from pyspark.mllib.recommendation import ALS, Rating

if len(sys.argv) != 5:
    print("Usage: ./bin/spark-submit Mansi_Ganatra_task1.py <train_file_path> <test_file_path> <case_id> <output_file_path>")
    exit(-1)
else:
    input_file_path_train = sys.argv[1]
    input_file_path_validation = sys.argv[2]
    case = int(sys.argv[3])
    output_file_path = sys.argv[4]

def process(entry):
    revisedEntries= entry[0].split(',')
    return (revisedEntries[0], revisedEntries[1], revisedEntries[2])

def convertValuesToTuple(entrySet):
    newEntrySet = []
    for entry in entrySet:
        newEntrySet += [(entry, 1)]
    return newEntrySet

def normalize_rating(rating, min_rating, max_min_diff, average):
    if rating == 0.0:
        return ((average - min_rating) / max_min_diff) * 4 + 1
    else:
        return((rating - min_rating) / max_min_diff) * 4 + 1

def model_based_CF(finalRdd_train, finalRdd_val):
    # variables to use in als model
    als_rank = 2
    als_seed = 5
    als_num_iterations = 20
    als_regularization_param = 0.2

    # users_business_train = finalRdd_train.map(lambda entry: (entry[0], entry[1])).sortByKey().collect()

    users_index_train = finalRdd_train.map(lambda entry: entry[0]).distinct().zipWithIndex().collectAsMap()
    businesses_index_train = finalRdd_train.map(lambda entry: entry[1]).distinct().zipWithIndex().collectAsMap()

    ratings_train = finalRdd_train.map(lambda entry: Rating(users_index_train[entry[0]], businesses_index_train[entry[1]], float(entry[2])))
    users_index_val = finalRdd_val.map(lambda entry: entry[0]).distinct().zipWithIndex().collectAsMap()
    users_index_val_reverse = finalRdd_val.map(lambda entry: entry[0]).distinct().zipWithIndex().map(
        lambda entry: (entry[1], entry[0])).collectAsMap()
    businesses_index_val = finalRdd_val.map(lambda entry: entry[1]).distinct().zipWithIndex().collectAsMap()
    businesses_index_val_reverse = finalRdd_val.map(lambda entry: entry[1]).distinct().zipWithIndex().map(
        lambda entry: (entry[1], entry[0])).collectAsMap()

    users_business_val = finalRdd_val.map(lambda entry: (users_index_val[entry[0]], businesses_index_val[entry[1]]))
    ratings_val = finalRdd_val.map(
        lambda entry: ((users_index_val[entry[0]], businesses_index_val[entry[1]]), float(entry[2])))

    # building als model using user ratings from train data
    als_model = ALS.train(ratings=ratings_train, iterations=als_num_iterations,
                          rank=als_rank, seed=als_seed, lambda_=als_regularization_param)

    # generating predictions from trained als model for ratings using validation/test data

    predicted_ratings = als_model.predictAll(users_business_val).map(
        lambda entry: ((entry[0], entry[1]), float(entry[2])))

    predicted_ratings_sum = predicted_ratings.values().sum()
    predicted_ratings_count = predicted_ratings.count()

    predicted_ratings_avg = float(predicted_ratings_sum)/predicted_ratings_count

    # normalize the predicted ratings for improving rmse as suggested here:https://stackoverflow.com/a/36762714
    # using min-max normalization strategy
    max_rating = predicted_ratings.map(lambda entry: entry[1]).max()
    min_rating = predicted_ratings.map(lambda entry: entry[1]).min()
    max_min_diff = max_rating - min_rating

    # print("Min: ", min_rating)
    # print("Max: ", max_rating)
    # print("Diff: ", max_min_diff)

    normalized_predicted_ratings = predicted_ratings\
        .mapValues(lambda entry: ((entry - min_rating) / max_min_diff) * 4 + 1)

    # normalized_predicted_ratings = predicted_ratings\
    #     .mapValues(lambda entry: normalize_rating(entry, min_rating, max_min_diff, predicted_ratings_avg))

    # Not required to handle missing values as the count came out to be zero
    # ratings_val_count = ratings_val.count()
    # normalized_predicted_ratings_count = normalized_predicted_ratings.count()
    # difference = ratings_val_count - normalized_predicted_ratings_count
    #
    # missing_values = ratings_val.subtract(predicted_ratings).collectAsMap()
    # # print("Missing: ", missing_values)
    #
    # print("Missing Value count: ", difference)


    # calculate rmse
    original_and_predicted = ratings_val.join(normalized_predicted_ratings)
    mean_squared_error = original_and_predicted.map(lambda entry: (entry[1][1] - entry[1][0]) ** 2).mean()
    rmse = math.sqrt(mean_squared_error)

    output_predictions = predicted_ratings \
        .map(lambda entry: (users_index_val_reverse[entry[0][0]], businesses_index_val_reverse[entry[0][1]], entry[1]))\
        .collect()

    with open(output_file_path, "w+", encoding="utf-8") as fp:
        fp.write("user_id,business_id,prediction")
        fp.write('\n')
        fp.write('\n'.join('{},{},{}'.format(x[0], x[1], x[2]) for x in output_predictions))

    print("RMSE: ", rmse)


def calculate_prediction_user_based_CF(user1, business):
    # Calculate weights for each user with other users using Pearson Correlation Coefficients

    # for each user-[business] calculate pearson weight for each user's co-rated users

    # predicted_ratings_list = []
    # users_co_rated_items_map = {}


    current_predicted_rating = 0.00
    current_predicted_rating_numerator = 0.00
    current_predicted_rating_denominator = 0.00
    pearson_weight_map = {}

    # in case no other user rated this item
    default_rating_tuple = average_rating_per_user.get(user1)

    default_rating = float(default_rating_tuple[0]) / default_rating_tuple[1]

    # find previously rated businesses by current user
    previous_businesses_of_current_user = set(users_business_train_map.get(user1))

    # find other users who rated the same business
    users_with_same_business = set()

    # for user2, all_businesses in users_business_train_map.items():
    #     if set(all_businesses).__contains__(business):
    #         users_with_same_business.add(user2)

    if business_user_train_map.get(business):
        users_with_same_business = set(business_user_train_map.get(business))

    users_to_remove = set()
    if len(users_with_same_business) != 0:

        # for each pair of user1, user2 calculate pearson weight coefficient of co-rated items
        for user2 in users_with_same_business:
            co_rated_businesses = set()

            user2_businesses = set(users_business_train_map.get(user2))
            co_rated_businesses = previous_businesses_of_current_user.intersection(user2_businesses)

            co_rated_businesses_length = len(co_rated_businesses)

            if not pearson_weight_map.keys().__contains__((user1, user2)) and co_rated_businesses_length != 0:
                user1_total_co_rated_ratings = 0.00
                user2_total_co_rated_ratings = 0.00

                pearson_weight = 0.00
                pearson_weight_numerator = 0.00
                pearson_weight_denominator = 0.00
                user1_denominator = 0.00
                user2_denominator = 0.00

                for co_rated_business in co_rated_businesses:
                    user1_total_co_rated_ratings += all_info_train_map.get((user1, co_rated_business))
                    user2_total_co_rated_ratings += all_info_train_map.get((user2, co_rated_business))

                user1_average_co_rated_rating = user1_total_co_rated_ratings / co_rated_businesses_length
                user2_average_co_rated_rating = user2_total_co_rated_ratings / co_rated_businesses_length

                for co_rated_business in co_rated_businesses:
                    user1_current_rating = all_info_train_map.get((user1, co_rated_business))
                    user2_current_rating = all_info_train_map.get((user2, co_rated_business))

                    user1_numerator = user1_current_rating - user1_average_co_rated_rating
                    user2_numerator = user2_current_rating - user2_average_co_rated_rating
                    pearson_weight_numerator += (user1_numerator * user2_numerator)

                    user1_denominator += (user1_numerator ** 2)
                    user2_denominator += (user2_numerator ** 2)

                pearson_weight_denominator = math.sqrt(user1_denominator) * math.sqrt(user2_denominator)

                if pearson_weight_numerator != 0 and pearson_weight_denominator != 0:
                    pearson_weight = abs(pearson_weight_numerator / pearson_weight_denominator)

                pearson_weight_map.update({tuple([user1, user2]): pearson_weight})

            else:
                users_to_remove.add(user2)

        # print("WEIGHT MAP: ", pearson_weight_map)
        # print("Previous count:", len(users_with_same_business))

        users_with_same_business -= users_to_remove
        # print("After removing count:", len(users_with_same_business))

        for user2_again in users_with_same_business:
            user2_pearson_weight = pearson_weight_map.get((user1, user2_again))

            if user2_pearson_weight != 0:
                # get the tuple from pre-calculated (sum,count)
                user2_average_tuple = average_rating_per_user.get(user2_again)

                # get current rating for current business
                user2_current_business = all_info_train_map.get((user2_again, business))

                # calculate average for all business except current business
                user2_average_rating_without_current_item = float(
                    user2_average_tuple[0] - user2_current_business) / (user2_average_tuple[1] - 1)

                # calculate weighted average
                user2_weighted_rating = (user2_current_business - user2_average_rating_without_current_item) \
                                        * user2_pearson_weight

                current_predicted_rating_numerator += user2_weighted_rating
                current_predicted_rating_denominator += user2_pearson_weight

        if current_predicted_rating_denominator != 0:
            current_predicted_rating = default_rating + float(
                current_predicted_rating_numerator) / current_predicted_rating_denominator

    else:
        current_predicted_rating = default_rating

    # predicted_ratings_list.append(((user1, business), current_predicted_rating))
    # print("Predicted for user, business, rating: ", ((user1, business), current_predicted_rating))
    return ((user1, business), current_predicted_rating)

def user_based_CF(finalRdd_train, finalRdd_val):

    ratings_val = finalRdd_val.map(
        lambda entry: ((entry[0], entry[1]), float(entry[2])))

    users_business_val_set = finalRdd_val.map(lambda entry: (entry[0], entry[1]))\
        .sortByKey()

    predicted_ratings = users_business_val_set\
        .map(lambda entry: calculate_prediction_user_based_CF(entry[0], entry[1]))\
        .persist()

    # normalize the predicted ratings for improving rmse as suggested here:https://stackoverflow.com/a/36762714
    # using min-max normalization strategy
    max_rating = predicted_ratings.map(lambda entry: entry[1]).max()
    min_rating = predicted_ratings.map(lambda entry: entry[1]).min()
    max_min_diff = max_rating - min_rating

    # print("Min: ", min_rating)
    # print("Max: ", max_rating)
    # print("Diff: ", max_min_diff)

    normalized_predicted_ratings = predicted_ratings\
        .mapValues(lambda entry: ((entry - min_rating) / max_min_diff) * 4 + 1.3)

    original_and_predicted = ratings_val.join(normalized_predicted_ratings)
    mean_squared_error = original_and_predicted.map(lambda entry: (entry[1][1] - entry[1][0]) ** 2).mean()
    rmse = math.sqrt(mean_squared_error)

    output_predictions = normalized_predicted_ratings \
        .map(lambda entry: (entry[0][0], entry[0][1], entry[1])) \
        .collect()

    # print("Completed Predicting")
    with open(output_file_path, "w+", encoding="utf-8") as fp:
        fp.write("user_id,business_id,prediction")
        fp.write('\n')
        fp.write('\n'.join('{},{},{}'.format(x[0], x[1], x[2]) for x in output_predictions))

    print("RMSE: ", rmse)


def calculate_prediction_item_based_CF(user, business1):
    # Calculate weights for each user with other users using Pearson Correlation Coefficients

    # for each user-[business] calculate pearson weight for each user's co-rated users

    # predicted_ratings_list = []
    # users_co_rated_items_map = {}

    # print("Current: ", user, business1)

    current_predicted_rating = 0.00
    current_predicted_rating_numerator = 0.00
    current_predicted_rating_denominator = 0.00
    pearson_weight_map = {}

    # in case no other user rated this item
    default_rating_tuple = average_rating_per_business.get(business1)

    # print("Default Rating: ", default_rating_tuple)
    default_rating = 1.0
    if default_rating_tuple is not None:
        default_rating = float(default_rating_tuple[0]) / default_rating_tuple[1]
    else:
        default_rating_tuple = average_rating_per_user.get(user)
        default_rating = float(default_rating_tuple[0]) / default_rating_tuple[1]

    # print("Default: ", default_rating)

    # find previously rated businesses by current user
    if business_user_train_map.get(business1):
        users_with_same_business = set(business_user_train_map.get(business1))
    else:
        users_with_same_business = set()

    previous_businesses_of_current_user = set(users_business_train_map.get(user))

    # find other users who rated the same business
    # users_with_same_business = set()

    # for user2, all_businesses in users_business_train_map.items():
    #     if set(all_businesses).__contains__(business):
    #         users_with_same_business.add(user2)

    # if business_user_train_map.get(business):
    #     users_with_same_business = set(business_user_train_map.get(business))

    businesses_to_remove = set()
    if len(users_with_same_business) != 0:

        # for each pair of user, user2 calculate pearson weight coefficient of co-rated items
        for business2 in previous_businesses_of_current_user:
            co_rated_users = set()

            business2_users = set(business_user_train_map.get(business2))
            co_rated_users = users_with_same_business.intersection(business2_users)

            co_rated_users_length = len(co_rated_users)

            if not pearson_weight_map.keys().__contains__((business1, business2)) and co_rated_users_length != 0:
                business1_total_co_rated_ratings = 0.00
                business2_total_co_rated_ratings = 0.00

                pearson_weight = 0.00
                pearson_weight_numerator = 0.00
                pearson_weight_denominator = 0.00
                business1_denominator = 0.00
                business2_denominator = 0.00

                for co_rated_user in co_rated_users:
                    business1_total_co_rated_ratings += all_info_train_map.get((co_rated_user,business1 ))
                    business2_total_co_rated_ratings += all_info_train_map.get((co_rated_user, business2))

                business1_average_co_rated_rating = business1_total_co_rated_ratings / co_rated_users_length
                business2_average_co_rated_rating = business2_total_co_rated_ratings / co_rated_users_length

                for co_rated_user in co_rated_users:
                    business1_current_rating = all_info_train_map.get((co_rated_user, business1))
                    business2_current_rating = all_info_train_map.get((co_rated_user, business2))

                    business1_numerator = business1_current_rating - business1_average_co_rated_rating
                    business2_numerator = business2_current_rating - business2_average_co_rated_rating
                    pearson_weight_numerator += (business1_numerator * business2_numerator)

                    business1_denominator += (business1_numerator ** 2)
                    business2_denominator += (business2_numerator ** 2)

                pearson_weight_denominator = math.sqrt(business1_denominator) * math.sqrt(business2_denominator)

                if pearson_weight_numerator != 0 and pearson_weight_denominator != 0:
                    pearson_weight = abs(pearson_weight_numerator / pearson_weight_denominator)

                pearson_weight_map.update({tuple([business1, business2]): pearson_weight})

            else:
                businesses_to_remove.add(business2)

        # print("WEIGHT MAP: ", pearson_weight_map)
        # print("Previous count:", len(users_with_same_business))

        previous_businesses_of_current_user -= businesses_to_remove
        # print("After removing count:", len(users_with_same_business))

        for business2_again in previous_businesses_of_current_user:
            business2_pearson_weight = pearson_weight_map.get((business1, business2_again))

            if business2_pearson_weight != 0:

                # get current rating for current business
                business2_current_user = all_info_train_map.get((user, business2_again))

                # calculate weighted average
                business2_weighted_rating = float(business2_current_user * business2_pearson_weight)
                current_predicted_rating_numerator += business2_weighted_rating
                current_predicted_rating_denominator += business2_pearson_weight

        if current_predicted_rating_denominator != 0:
            current_predicted_rating = float(
                current_predicted_rating_numerator) / current_predicted_rating_denominator
        if current_predicted_rating == 0:
            current_predicted_rating = default_rating

    else:
        current_predicted_rating = default_rating

    # predicted_ratings_list.append(((user, business), current_predicted_rating))
    # print("Predicted for user, business, rating: ", ((user, business1), current_predicted_rating))
    return ((user, business1), current_predicted_rating)


def item_based_CF(finalRdd_train, finalRdd_val):
    ratings_val = finalRdd_val.map(
        lambda entry: ((entry[0], entry[1]), float(entry[2])))

    users_business_val_set = finalRdd_val.map(lambda entry: (entry[0], entry[1]))\
        .sortByKey()

    predicted_ratings = users_business_val_set\
        .map(lambda entry: calculate_prediction_item_based_CF(entry[0], entry[1]))\
        .persist()

    # normalize the predicted ratings for improving rmse as suggested here:https://stackoverflow.com/a/36762714
    # using min-max normalization strategy
    max_rating = predicted_ratings.map(lambda entry: entry[1]).max()
    min_rating = predicted_ratings.map(lambda entry: entry[1]).min()
    max_min_diff = max_rating - min_rating

    # print("Min: ", min_rating)
    # print("Max: ", max_rating)
    # print("Diff: ", max_min_diff)

    normalized_predicted_ratings = predicted_ratings\
        .mapValues(lambda entry: ((entry - min_rating) / max_min_diff) * 4 + 1)

    original_and_predicted = ratings_val.join(normalized_predicted_ratings)
    mean_squared_error = original_and_predicted.map(lambda entry: (entry[1][1] - entry[1][0]) ** 2).mean()
    rmse = math.sqrt(mean_squared_error)

    output_predictions = original_and_predicted \
        .map(lambda entry: (entry[0][0], entry[0][1], entry[1][0], entry[1][1])) \
        .collect()

    # print("Completed Predicting")
    with open(output_file_path, "w+", encoding="utf-8") as fp:
        fp.write("user_id,business_id,original,prediction")
        fp.write('\n')
        fp.write('\n'.join('{},{},{},{}'.format(x[0], x[1], x[2], x[3]) for x in output_predictions))

    print("RMSE: ", rmse)


def generate_minhash_Array(users,num_hashes, num_users):
    users = list(users)
    system_max_value = sys.maxsize
    hashed_users = [system_max_value for i in range(0,num_hashes)]
    # random_a = list(random_coeffs.get('a'))
    # random_b = list(random_coeffs.get('b'))

    for user in users:
        for i in range(1, num_hashes+1):
            current_hash_code = ((i*user)+ (5*i*13)) % num_users

            if current_hash_code < hashed_users[i-1]:
                hashed_users[i-1] = current_hash_code
    return hashed_users

def applyLSHToSignature(business_id, signatures, n_bands, n_rows):
    signature_tuples = []
    for band in range(0,n_bands):
        band_name = band
        final_signature = signatures[band*n_rows:(band*n_rows)+n_rows]
        # print(final_signature)
        final_signature.insert(0, band_name)
        # print(final_signature)
        # print(str(final_signature))
        signature_tuple = (tuple(final_signature), business_id)
        signature_tuples.append(signature_tuple)

    return signature_tuples

def generate_similar_businesses(businesses):

    b_length = len(businesses)

    similar_businesses = []

    # if b_length == 2:
    #     similar_businesses.append((businesses[0], businesses[1]))
    # else:
    similar_businesses = sorted(list(itertools.combinations(sorted(businesses),2)))

    return similar_businesses

def calculate_jaccard(candidate, business_char_matrix):

    users_c1 = set(business_char_matrix.get(candidate[0]))
    users_c2 = set(business_char_matrix.get(candidate[1]))

    jaccard_intersection = len(users_c1.intersection(users_c2))
    jaccard_union = len(users_c1.union(users_c2))

    jaccard_similarity_value = float(jaccard_intersection)/float(jaccard_union)

    return (candidate, jaccard_similarity_value)

def generate_jaccard_pairs(finalRdd):

    jaccard_support = 0.5
    num_hashes = 80
    n_bands = 40
    n_rows = 2
    users = finalRdd.map(lambda entry: entry[0]).distinct()
    num_users = users.count()
    # businesses = finalRdd.map(lambda entry: entry[1]).distinct()
    # num_businesses = businesses.count()

    # because we are calculating similar businesses
    user_index_dict = finalRdd.map(lambda entry: entry[0]).zipWithIndex().collectAsMap()
    # user_index_reverse_dict = finalRdd.map(lambda entry: entry[0]).zipWithIndex().map(lambda entry: (entry[1], entry[0])).collectAsMap()
    # business_index_dict = finalRdd.map(lambda entry: entry[1]).zipWithIndex().collectAsMap()
    # business_index_reverse_dict = finalRdd.map(lambda entry: entry[1]).zipWithIndex().map(lambda entry: (entry[1], entry[0])).collectAsMap()

    business_user_map = finalRdd \
        .map(lambda entry: (entry[1], user_index_dict.get(entry[0]))) \
        .groupByKey() \
        .sortByKey() \
        .mapValues(lambda entry: set(entry)) \
        .persist()

    business_user_map_values = business_user_map.collect()
    business_char_matrix = {}

    for bu in business_user_map_values:
        business_char_matrix.update({bu[0]: bu[1]})

    # random_coeffs = dict(sc.broadcast(generateRandomHashCoefficients(num_hashes, num_users)).value)

    candidates = business_user_map \
        .mapValues(lambda users: generate_minhash_Array(users, num_hashes, num_users)) \
        .flatMap(lambda entry: applyLSHToSignature(entry[0], list(entry[1]), n_bands, n_rows)) \
        .groupByKey() \
        .filter(lambda entry: len(list(entry[1])) > 1) \
        .flatMap(lambda entry: generate_similar_businesses(sorted(list(entry[1])))) \
        .distinct() \
        .persist()

    final_pairs = candidates \
        .map(lambda cd: calculate_jaccard(cd, business_char_matrix)) \
        .filter(lambda cd: cd[1] >= jaccard_support)\
        .map(lambda cd: cd[0])\
        .sortByKey()

    return final_pairs.collect()

def calculate_prediction_item_based_CF_with_jaccard(user, business1, jaccard_items_map):
    # Calculate weights for each user with other users using Pearson Correlation Coefficients

    # for each user-[business] calculate pearson weight for each user's co-rated users

    # predicted_ratings_list = []
    # users_co_rated_items_map = {}

    # print("Current: ", user, business1)

    current_predicted_rating = 0.00
    current_predicted_rating_numerator = 0.00
    current_predicted_rating_denominator = 0.00
    pearson_weight_map = {}

    # in case no other user rated this item
    default_rating_tuple = average_rating_per_business.get(business1)

    # print("Default Rating: ", default_rating_tuple)
    default_rating = 1.0
    if default_rating_tuple is not None:
        default_rating = float(default_rating_tuple[0]) / default_rating_tuple[1]
    else:
        default_rating_tuple = average_rating_per_user.get(user)
        default_rating = float(default_rating_tuple[0]) / default_rating_tuple[1]

    # print("Default: ", default_rating)

    # find previously rated businesses by current user
    if business_user_train_map.get(business1):
        users_with_same_business = set(business_user_train_map.get(business1))
    else:
        users_with_same_business = set()

    previous_businesses_of_current_user = set(users_business_train_map.get(user))

    similar_businesses_from_jaccard = jaccard_items_map.get(business1)
    if similar_businesses_from_jaccard is not None:
        previous_businesses_of_current_user.union(similar_businesses_from_jaccard)

    # find other users who rated the same business
    # users_with_same_business = set()

    # for user2, all_businesses in users_business_train_map.items():
    #     if set(all_businesses).__contains__(business):
    #         users_with_same_business.add(user2)

    # if business_user_train_map.get(business):
    #     users_with_same_business = set(business_user_train_map.get(business))

    businesses_to_remove = set()
    if len(users_with_same_business) != 0:

        # for each pair of user, user2 calculate pearson weight coefficient of co-rated items
        for business2 in previous_businesses_of_current_user:
            co_rated_users = set()

            business2_users = set(business_user_train_map.get(business2))
            co_rated_users = users_with_same_business.intersection(business2_users)

            co_rated_users_length = len(co_rated_users)

            if not pearson_weight_map.keys().__contains__((business1, business2)) and co_rated_users_length != 0:
                business1_total_co_rated_ratings = 0.00
                business2_total_co_rated_ratings = 0.00

                pearson_weight = 0.00
                pearson_weight_numerator = 0.00
                pearson_weight_denominator = 0.00
                business1_denominator = 0.00
                business2_denominator = 0.00

                for co_rated_user in co_rated_users:
                    business1_total_co_rated_ratings += all_info_train_map.get((co_rated_user,business1 ))
                    business2_total_co_rated_ratings += all_info_train_map.get((co_rated_user, business2))

                business1_average_co_rated_rating = business1_total_co_rated_ratings / co_rated_users_length
                business2_average_co_rated_rating = business2_total_co_rated_ratings / co_rated_users_length

                for co_rated_user in co_rated_users:
                    business1_current_rating = all_info_train_map.get((co_rated_user, business1))
                    business2_current_rating = all_info_train_map.get((co_rated_user, business2))

                    business1_numerator = business1_current_rating - business1_average_co_rated_rating
                    business2_numerator = business2_current_rating - business2_average_co_rated_rating
                    pearson_weight_numerator += (business1_numerator * business2_numerator)

                    business1_denominator += (business1_numerator ** 2)
                    business2_denominator += (business2_numerator ** 2)

                pearson_weight_denominator = math.sqrt(business1_denominator) * math.sqrt(business2_denominator)

                if pearson_weight_numerator != 0 and pearson_weight_denominator != 0:
                    pearson_weight = abs(pearson_weight_numerator / pearson_weight_denominator)

                pearson_weight_map.update({tuple([business1, business2]): pearson_weight})

            else:
                businesses_to_remove.add(business2)

        # print("WEIGHT MAP: ", pearson_weight_map)
        # print("Previous count:", len(users_with_same_business))

        previous_businesses_of_current_user -= businesses_to_remove
        # print("After removing count:", len(users_with_same_business))

        for business2_again in previous_businesses_of_current_user:
            business2_pearson_weight = pearson_weight_map.get((business1, business2_again))

            if business2_pearson_weight != 0:

                # get current rating for current business
                business2_current_user = all_info_train_map.get((user, business2_again))

                # calculate weighted average
                business2_weighted_rating = float(business2_current_user * business2_pearson_weight)
                current_predicted_rating_numerator += business2_weighted_rating
                current_predicted_rating_denominator += business2_pearson_weight

        if current_predicted_rating_denominator != 0:
            current_predicted_rating = float(
                current_predicted_rating_numerator) / current_predicted_rating_denominator
        if current_predicted_rating == 0:
            current_predicted_rating = default_rating

    else:
        current_predicted_rating = default_rating

    # predicted_ratings_list.append(((user, business), current_predicted_rating))
    # print("Predicted for user, business, rating: ", ((user, business1), current_predicted_rating))
    return ((user, business1), current_predicted_rating)

def item_based_CF_with_jaccard(finalRdd_train, finalRdd_val):

    ratings_val = finalRdd_val.map(
        lambda entry: ((entry[0], entry[1]), float(entry[2])))

    users_business_val_set = finalRdd_val.map(lambda entry: (entry[0], entry[1]))\
        .sortByKey()

    predicted_ratings = users_business_val_set\
        .map(lambda entry: calculate_prediction_item_based_CF_with_jaccard(entry[0], entry[1], jaccard_items_map))\
        .persist()

    # normalize the predicted ratings for improving rmse as suggested here:https://stackoverflow.com/a/36762714
    # using min-max normalization strategy
    max_rating = predicted_ratings.map(lambda entry: entry[1]).max()
    min_rating = predicted_ratings.map(lambda entry: entry[1]).min()
    max_min_diff = max_rating - min_rating

    # print("Min: ", min_rating)
    # print("Max: ", max_rating)
    # print("Diff: ", max_min_diff)

    normalized_predicted_ratings = predicted_ratings\
        .mapValues(lambda entry: ((entry - min_rating) / max_min_diff) * 4 + 1)

    original_and_predicted = ratings_val.join(normalized_predicted_ratings)
    mean_squared_error = original_and_predicted.map(lambda entry: (entry[1][1] - entry[1][0]) ** 2).mean()
    rmse = math.sqrt(mean_squared_error)

    output_predictions = original_and_predicted \
        .map(lambda entry: (entry[0][0], entry[0][1], entry[1][0], entry[1][1])) \
        .collect()

    # print("Completed Predicting")
    with open(output_file_path, "w+", encoding="utf-8") as fp:
        fp.write("user_id,business_id,original,prediction")
        fp.write('\n')
        fp.write('\n'.join('{},{},{},{}'.format(x[0], x[1], x[2], x[3]) for x in output_predictions))

    print("RMSE: ", rmse)

# main function starts here
result = []
SparkContext.setSystemProperty('spark.executor.memory', '4g')
SparkContext.setSystemProperty('spark.driver.memory', '4g')
sc = SparkContext('local[*]', 'task2')

# input_file_path_validation = "./yelp_val.csv"
# input_file_path_train = "./yelp_train.csv"
# output_file_path = "./task2__result_4.csv"
# case = 4

start = time()

# loading and parsing training data
user_businessRdd_train = sc.textFile(input_file_path_train).map(lambda entry: entry.split('\n')).map(lambda entry: process(entry))
headers_train = user_businessRdd_train.take(1)
finalRdd_train = user_businessRdd_train.filter(lambda entry: entry[0] != headers_train[0][0]).persist()

# loading and parsing test data
user_businessRdd_val = sc.textFile(input_file_path_validation).map(lambda entry: entry.split('\n')).map(lambda entry: process(entry))
headers_val = user_businessRdd_val.take(1)
finalRdd_val = user_businessRdd_val.filter(lambda entry: entry[0] != headers_val[0][0]).persist()

if case == 1:
    model_based_CF(finalRdd_train, finalRdd_val)

elif case == 2:
    all_info_train_map = sc.broadcast(
        finalRdd_train.map(lambda entry: ((entry[0], entry[1]), float(entry[2]))).collectAsMap()).value
    users_business_train_map_rdd = finalRdd_train.map(lambda entry: (entry[0], entry[1])) \
        .groupByKey() \
        .mapValues(lambda entry: set(entry)) \
        .sortByKey() \
        .collectAsMap()
    users_business_train_map = sc.broadcast(users_business_train_map_rdd).value

    business_user_train_map_rdd = finalRdd_train.map(lambda entry: (entry[1], entry[0])) \
        .groupByKey() \
        .mapValues(lambda entry: set(entry)) \
        .sortByKey() \
        .collectAsMap()

    business_user_train_map = sc.broadcast(business_user_train_map_rdd).value

    average_rating_per_user_rdd = finalRdd_train.map(lambda entry: (entry[0], float(entry[2]))) \
        .mapValues(lambda entry: (entry, 1)) \
        .reduceByKey(lambda e1, e2: (e1[0] + e2[0], e1[1] + e2[1])) \
        .sortByKey() \
        .collectAsMap()

    average_rating_per_user = sc.broadcast(average_rating_per_user_rdd).value

    user_based_CF(finalRdd_train, finalRdd_val)

elif case == 3:
    all_info_train_map = sc.broadcast(
        finalRdd_train.map(lambda entry: ((entry[0], entry[1]), float(entry[2]))).collectAsMap()).value

    users_business_train_map_rdd = finalRdd_train.map(lambda entry: (entry[0], entry[1])) \
        .groupByKey() \
        .mapValues(lambda entry: set(entry)) \
        .sortByKey() \
        .collectAsMap()
    users_business_train_map = sc.broadcast(users_business_train_map_rdd).value

    business_user_train_map_rdd = finalRdd_train.map(lambda entry: (entry[1], entry[0])) \
        .groupByKey() \
        .mapValues(lambda entry: set(entry)) \
        .sortByKey() \
        .collectAsMap()

    business_user_train_map = sc.broadcast(business_user_train_map_rdd).value
    # print(business_user_train_map)

    average_rating_per_business_rdd = finalRdd_train.map(lambda entry: (entry[1], float(entry[2]))) \
        .mapValues(lambda entry: (entry, 1)) \
        .reduceByKey(lambda e1, e2: (e1[0] + e2[0], e1[1] + e2[1])) \
        .sortByKey() \
        .collectAsMap()

    average_rating_per_business = sc.broadcast(average_rating_per_business_rdd).value

    average_rating_per_user_rdd = finalRdd_train.map(lambda entry: (entry[0], float(entry[2]))) \
        .mapValues(lambda entry: (entry, 1)) \
        .reduceByKey(lambda e1, e2: (e1[0] + e2[0], e1[1] + e2[1])) \
        .sortByKey() \
        .collectAsMap()

    average_rating_per_user = sc.broadcast(average_rating_per_user_rdd).value

    # print(average_rating_per_business)

    item_based_CF(finalRdd_train, finalRdd_val)

elif case ==4:
    jaccard_items = generate_jaccard_pairs(finalRdd_train)
    jaccard_items_map_sc = {}

    for pair in jaccard_items:
        business1 = pair[0]
        business2 = pair[1]

        if business1 in jaccard_items_map_sc.keys():
            jaccard_items_map_sc[business1].add(business2)
        else:
            jaccard_items_map_sc[business1] = set([business2])

        if business2 in jaccard_items_map_sc:
            jaccard_items_map_sc[business2].add(business1)
        else:
            jaccard_items_map_sc[business2] = set([business1])

    jaccard_items_map = dict(sc.broadcast(jaccard_items_map_sc).value)
    all_info_train_map = sc.broadcast(
        finalRdd_train.map(lambda entry: ((entry[0], entry[1]), float(entry[2]))).collectAsMap()).value

    users_business_train_map_rdd = finalRdd_train.map(lambda entry: (entry[0], entry[1])) \
        .groupByKey() \
        .mapValues(lambda entry: set(entry)) \
        .sortByKey() \
        .collectAsMap()
    users_business_train_map = sc.broadcast(users_business_train_map_rdd).value

    business_user_train_map_rdd = finalRdd_train.map(lambda entry: (entry[1], entry[0])) \
        .groupByKey() \
        .mapValues(lambda entry: set(entry)) \
        .sortByKey() \
        .collectAsMap()

    business_user_train_map = sc.broadcast(business_user_train_map_rdd).value
    # print(business_user_train_map)

    average_rating_per_business_rdd = finalRdd_train.map(lambda entry: (entry[1], float(entry[2]))) \
        .mapValues(lambda entry: (entry, 1)) \
        .reduceByKey(lambda e1, e2: (e1[0] + e2[0], e1[1] + e2[1])) \
        .sortByKey() \
        .collectAsMap()

    average_rating_per_business = sc.broadcast(average_rating_per_business_rdd).value

    average_rating_per_user_rdd = finalRdd_train.map(lambda entry: (entry[0], float(entry[2]))) \
        .mapValues(lambda entry: (entry, 1)) \
        .reduceByKey(lambda e1, e2: (e1[0] + e2[0], e1[1] + e2[1])) \
        .sortByKey() \
        .collectAsMap()

    average_rating_per_user = sc.broadcast(average_rating_per_user_rdd).value

    # print(average_rating_per_business)

    item_based_CF_with_jaccard(finalRdd_train, finalRdd_val)

end = time()
# als_model.save(sc, "model.txt")
print("Duration: ", end-start)