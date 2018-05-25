# Ruby 2.0
# Reads stdin: ruby -n preprocess-twitter.rb
#
# Script for preprocessing tweets by Romain Paulus
# with small modifications by Jeffrey Pennington

def tokenize input

	# Different regex parts for smiley faces
	eyes = "[8:=;]"
	nose = "['`\-]?"

	input = input
		.gsub(/https?:\/\/\S+\b|www\.(\w+\.)+\S*/,"<url>")
		.gsub("/"," / ") 			# Force splitting words appended with slashes (once we tokenized the URLs, of course)
		.gsub(/@\w+/, "<user>")		# User is mentioned with '@' symbol
		.gsub(/#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}/i, "<smile>")	# smiling face detected
		.gsub(/#{eyes}#{nose}p+/i, "<lolface>")
		.gsub(/#{eyes}#{nose}\(+|\)+#{nose}#{eyes}/, "<sadface>")
		.gsub(/#{eyes}#{nose}[\/|l*]/, "<neutralface>")
		.gsub(/<3/,"<heart>")
		.gsub(/[-+]?[.\d]*[\d]+[:,.\d]*/, "<number>")
		.gsub(/#\S+/){ |hashtag| # Split hashtags on uppercase letters

			hashtag_body = hashtag[1..-1]
			result = (["<hashtag>"] + hashtag_body.split(/(?=[A-Z])/)).join(" ")
			result
		} 
		.gsub(/([!?.]){2,}/){ # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
			"#{$~[1]} <repeat>"
		}
		.gsub(/\b(\S*?)(.)\2{2,}\b/){ # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
			$~[1] + $~[2] + " <elong>"
		}

	return input
end

puts tokenize($_)
