import java.util.*;

/**
 * 80道算法题
 */
public class AQ80 {
    //1大数加法
    public String t1_bigNumberAdd(String s, String t) {
        // write code here
        Stack<Integer> stack = new Stack<>();
        StringBuilder result = new StringBuilder();
        int i = s.length() - 1;
        int j = t.length() - 1;
        int carry = 0;
        while (i >= 0 || j >= 0 || carry != 0) {
            carry += i >= 0 ? s.charAt(i--) - '0' : 0;
            carry += j >= 0 ? t.charAt(j--) - '0' : 0;
            stack.push(carry % 10);
            carry /= 10;
        }
        while (!stack.isEmpty()) {
            result.append(stack.pop());
        }
        return result.toString();
    }

    //2链表中环的入口结点
    public static class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public ListNode t2_entryNodeOfLoop(ListNode pHead) {
        if (pHead == null) return null;
        ListNode slow = pHead;
        ListNode fast = pHead;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                break;
            }
        }

        if (fast == null || fast.next == null) {
            return null;
        }

        fast = pHead;

        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }

        return fast;

    }

    //3判断链表中是否有环
    public boolean t3_hasCycle(ListNode head) {
        ListNode pos = head;
        Set<ListNode> visited = new HashSet<>();
        while (pos != null) {
            if (visited.contains(pos)) {
                return true;
            } else {
                visited.add(pos);
            }
            pos = pos.next;
        }
        return false;
    }

    //4二叉树中的最大路径和
    public static class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int i) {
            val = i;
        }
    }

    int maxSum = Integer.MIN_VALUE;

    public int t4_maxPathSum(TreeNode root) {
        // write code here
        maxGain(root);
        return maxSum;
    }

    public int maxGain(TreeNode node) {
        if (node == null) {
            return 0;
        }

        int leftGain = Math.max(maxGain(node.left), 0);
        int rightGain = Math.max(maxGain(node.right), 0);

        int priceNewPath = node.val + leftGain + rightGain;
        maxSum = Math.max(maxSum, priceNewPath);

        return node.val + Math.max(leftGain, rightGain);
    }

    //5将升序数组转化为平衡二叉搜索树
    public TreeNode t5_sortedArrayToBST(int[] num) {
        // write code here
        if (num == null || num.length == 0) {
            return null;
        }

        return process(num, 0, num.length - 1);
    }

    public TreeNode process(int[] num, int left, int right) {
        if (left > right) {
            return null;
        }

        if (left == right) {
            return new TreeNode(num[left]);
        }

        int len = right - left + 1;
        int mid = left + len / 2;
        TreeNode root = new TreeNode(num[mid]);
        root.left = process(num, left, mid - 1);
        root.right = process(num, mid + 1, right);
        return root;
    }

    //6重建二叉树
    public TreeNode t6_reConstructBinaryTree(int[] pre, int[] vin) {
        return reConstructBinaryTree(pre, 0, pre.length - 1, vin, 0, vin.length - 1);
    }

    private TreeNode reConstructBinaryTree(int[] pre, int startPre, int endPre, int[] vin, int startVin, int endVin) {
        if (startPre > endPre || startVin > endVin) {
            return null;
        }

        TreeNode root = new TreeNode(pre[startPre]);

        for (int i = startVin; i <= endVin; i++) {
            if (vin[i] == pre[startPre]) {
                root.left = reConstructBinaryTree(pre, startPre + 1, startPre + i - startVin, vin, startVin, i - 1);
                root.right = reConstructBinaryTree(pre, i - startVin + startPre + 1, endPre, vin, i + 1, endVin);
                break;
            }
        }

        return root;
    }

    //7按之字形顺序打印二叉树
    public ArrayList<ArrayList<Integer>> t7_print(TreeNode pRoot) {
        LinkedList<TreeNode> deque = new LinkedList<>();
        ArrayList res = new ArrayList<>();
        if (pRoot != null) {
            deque.add(pRoot);
        }
        while (!deque.isEmpty()) {
            ArrayList tmp = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                TreeNode node = deque.removeFirst();
                tmp.add(node.val);
                if (node.left != null) {
                    deque.addLast(node.left);
                }
                if (node.right != null) {
                    deque.addLast(node.right);
                }
            }
            res.add(tmp);
            if (deque.isEmpty()) {
                break;
            }
            tmp = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                TreeNode node = deque.removeLast();
                tmp.add(node.val);
                if (node.right != null) {
                    deque.addFirst(node.right);
                }
                if (node.left != null) {
                    deque.addFirst(node.left);
                }
            }
            res.add(tmp);
        }

        return res;
    }

    //8求二叉树的层序遍历
    public ArrayList<ArrayList<Integer>> t8_levelOrder(TreeNode root) {
        // write code here
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        dfs(root, res, 0);
        return res;
    }

    public void dfs(TreeNode root, ArrayList<ArrayList<Integer>> list, int depth) {
        if (root == null) {
            return;
        }

        if (list.size() == depth) {
            list.add(new ArrayList<>());
        }

        dfs(root.left, list, depth + 1);
        list.get(depth).add(root.val);
        dfs(root.right, list, depth + 1);
    }

    //9对称的二叉树
    boolean t9_isSymmetrical(TreeNode pRoot) {
        return isSame(pRoot, pRoot);
    }

    public boolean isSame(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null || root2 == null) {
            return false;
        }
        return root1.val == root2.val &&
                isSame(root1.left, root2.right) &&
                isSame(root1.right, root2.left);
    }

    //10最长回文子串
    public int t10_getLongestPalindrome(String A) {
        // write code here
        int maxLen = 0;
        int n = A.length();

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                String now = A.substring(i, j);
                if (isHuiWen(now) && now.length() > maxLen) {
                    maxLen = now.length();
                }
            }
        }

        return maxLen;
    }

    public boolean isHuiWen(String s) {
        int len = s.length();
        for (int i = 0; i < len / 2; i++) {
            if (s.charAt(i) != s.charAt(len - i - 1)) {
                return false;
            }
        }
        return true;
    }

    //11顺时针旋转矩阵
    public int[][] t11_rotateMatrix(int[][] mat, int n) {
        // write code here
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int tmp = mat[i][j];
                mat[i][j] = mat[j][i];
                mat[j][i] = tmp;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                int tmp = mat[i][n - 1 - j];
                mat[i][n - 1 - j] = mat[i][j];
                mat[i][j] = tmp;
            }
        }

        return mat;
    }

    //12连续子数组的最大和
    public int t12_findGreatestSumOfSubArray(int[] array) {
        int res = array[0];
        int max = array[0];
        for (int i = 1; i < array.length; i++) {
            max = Math.max(max + array[i], array[i]);
            res = Math.max(max, res);
        }
        return res;
    }

    //13合并两个有序的数组
    public void t13_merge(int A[], int m, int B[], int n) {
        int i = 0, j = 0, p = 0;
        int[] c = new int[m + n];
        while (i < m && j < n) {
            c[p++] = A[i] <= B[j] ? A[i++] : B[j++];
        }
        while (i < m) {
            c[p++] = A[i++];
        }
        while (j < n) {
            c[p++] = B[j++];
        }
        for (int x = 0; x < p; x++) {
            A[x] = c[x];
        }
    }

    //14删除有序链表中重复的元素-I
    public ListNode t14_deleteDuplicates(ListNode head) {
        // write code here
        ListNode cur = head;
        while (cur != null) {
            while (cur.next != null && cur.val == cur.next.val) {
                cur.next = cur.next.next;
            }
            cur = cur.next;
        }
        return head;
    }

    //15删除有序链表中重复的元素-II
    public ListNode t15_deleteDuplicates(ListNode head) {
        // write code here
        ListNode node = new ListNode(-1);
        node.next = head;
        ListNode p = node;
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            if (cur.val != cur.next.val) {
                p = cur;
            } else {
                while (cur.next != null && cur.val == cur.next.val) {
                    cur = cur.next;

                }
                p.next = cur.next;

            }
            cur = cur.next;
        }
        return node.next;
    }

    //16括号生成
    public ArrayList<String> t16_generateParenthesis(int n) {
        // write code here
        ArrayList<String> result = new ArrayList<>(10);
        backtrack("", 0, 0, n, result);
        return result;
    }

    private void backtrack(String string, int open, int close, int n, List<String> result) {
        if (string.length() == n << 1) {
            result.add(string);
            return;
        }
        if (open < n) {
            backtrack(string + "(", open + 1, close, n, result);
        }
        if (close < open) {
            backtrack(string + ")", open, close + 1, n, result);
        }
    }

    //17集合的所有子集(一)
    ArrayList<ArrayList<Integer>> result = new ArrayList<>();

    public ArrayList<ArrayList<Integer>> t17_subsets(int[] nums) {
        ArrayList<Integer> list = new ArrayList<>();
        Arrays.sort(nums);
        for (int j = 0; j <= nums.length; j++) {
            backtracking(nums, j, 0, list);
        }
        return result;
    }

    public void backtracking(int[] nums, int k, int start, ArrayList<Integer> list) {
        if (k < 0) {
            return;
        } else if (k == 0) {
            result.add(new ArrayList(list));
        } else {
            for (int i = start; i < nums.length; i++) {
                list.add(nums[i]);
                backtracking(nums, k - 1, i + 1, list);
                list.remove(list.size() - 1);
            }
        }
    }

    //18最小覆盖子串
    public String t18_minWindow(String S, String T) {
        // write code here
        int[] map = new int[128];
        for (int i = 0; i < T.length(); i++) {
            map[T.charAt(i)]++;
        }

        int begin = 0, end = 0, d = Integer.MAX_VALUE, counter = T.length(), head = 0;
        while (end < S.length()) {
            if (map[S.charAt(end++)]-- > 0) {
                counter--;
            }
            while (counter == 0) {
                if (end - begin < d) {
                    d = end - (head = begin);
                }
                if (map[S.charAt(begin++)]++ == 0) {
                    counter++;
                }
            }
        }
        return d == Integer.MAX_VALUE ? "" : S.substring(head, head + d);
    }

    //19缺失的第一个正整数
    public int t19_minNumberDisappeared(int[] nums) {
        // write code here
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= 0) {
                nums[i] = nums.length + 1;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            int x = Math.abs(nums[i]);
            if (x <= nums.length) {
                nums[x - 1] = (-1) * Math.abs(nums[x - 1]);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return nums.length + 1;
    }

    //20第一个只出现一次的字符
    public int t20_FirstNotRepeatingChar(String str) {
        int[] dp = new int[123];
        for (int i = 0; i < str.length(); i++) {
            dp[str.charAt(i)]++;
        }
        for (int i = 0; i < str.length(); i++) {
            if (dp[str.charAt(i)] == 1) {
                return i;
            }
        }
        return -1;
    }

    //21合并两个排序的链表
    public ListNode t21_Merge(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        if (list1.val <= list2.val) {
            list1.next = t21_Merge(list1.next, list2);
            return list1;
        } else {
            list2.next = t21_Merge(list1, list2.next);
            return list2;
        }
    }

    //22编辑距离(二)
    public int t22_minEditCost(String str1, String str2, int ic, int dc, int rc) {
        // write code here
        if (str1.length() == 0) {
            return str2.length() * ic;
        }
        if (str2.length() == 0) {
            return str1.length() * dc;
        }
        int n1 = str1.length(), n2 = str2.length();
        int[][] dp = new int[n1 + 1][n2 + 1];
        for (int i = 0; i <= n1; i++) {
            dp[i][0] = i * dc;
        }
        for (int i = 0; i <= n2; i++) {
            dp[0][i] = i * ic;
        }
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j] + dc, Math.min(dp[i][j - 1] + ic, dp[i - 1][j - 1] + rc));
                }
            }
        }
        return dp[n1][n2];
    }

    //23在两个长度相等的排序数组中找到上中位数
    public int t23_findMedianinTwoSortedAray(int[] arr1, int[] arr2) {
        // write code here
        if (arr1 == null || arr2 == null || arr1.length != arr2.length) {
            return 0;
        }
        int left1 = 0;
        int right1 = arr1.length - 1;
        int left2 = 0;
        int right2 = arr2.length - 1;
        int mid1 = 0;
        int mid2 = 0;
        int offset = 0;
        while (left1 < right1) {
            mid1 = left1 + (right1 - left1) / 2;
            mid2 = left2 + (right2 - left2) / 2;
            offset = ((right1 - left1 + 1) & 1) ^ 1;
            if (arr1[mid1] > arr2[mid2]) {
                right1 = mid1;
                left2 = mid2 + offset;
            } else if (arr1[mid1] < arr2[mid2]) {
                right2 = mid2;
                left1 = mid1 + offset;
            } else {
                return arr1[mid1];
            }
        }
        return Math.min(arr1[left1], arr2[left2]);
    }

    //24合并区间
    public class Interval {
        int start;
        int end;

        Interval() {
            start = 0;
            end = 0;
        }

        Interval(int s, int e) {
            start = s;
            end = e;
        }
    }

    public ArrayList<Interval> t24_merge(ArrayList<Interval> intervals) {
        ArrayList<Interval> res = new ArrayList<>();
        Collections.sort(intervals, (a, b) -> a.start - b.start);
        int len = intervals.size();
        int i = 0;
        while (i < len) {
            int left = intervals.get(i).start;
            int right = intervals.get(i).end;
            while (i < len - 1 && intervals.get(i + 1).start <= right) {
                right = Math.max(right, intervals.get(i + 1).end);
                i++;
            }
            res.add(new Interval(left, right));
            i++;
        }
        return res;
    }

    //25两个链表生成相加链表
    public ListNode t25_addInList(ListNode head1, ListNode head2) {
        // write code here
        if (head1 == null) {
            return head2;
        }
        if (head2 == null) {
            return head1;
        }
        head1 = reverse(head1);
        head2 = reverse(head2);
        ListNode head = new ListNode(-1);
        ListNode nHead = head;
        int tmp = 0;
        while (head1 != null || head2 != null) {
            int val = tmp;
            if (head1 != null) {
                val += head1.val;
                head1 = head1.next;
            }
            if (head2 != null) {
                val += head2.val;
                head2 = head2.next;
            }
            tmp = val / 10;
            nHead.next = new ListNode(val % 10);
            nHead = nHead.next;
        }
        if (tmp > 0) {
            nHead.next = new ListNode(tmp);
        }
        return reverse(head.next);
    }

    ListNode reverse(ListNode head) {
        if (head == null) {
            return head;
        }
        ListNode cur = head;
        ListNode node = null;
        while (cur != null) {
            ListNode tail = cur.next;
            cur.next = node;
            node = cur;
            cur = tail;
        }
        return node;
    }

    //26最长无重复子数组
    public int t26_maxLength(int[] arr) {
        // write code here
        if (arr.length == 0) {
            return 0;
        }
        HashMap<Integer, Integer> map = new HashMap<>();
        int max = 0;
        for (int i = 0, j = 0; i < arr.length; ++i) {
            if (map.containsKey(arr[i])) {
                j = Math.max(j, map.get(arr[i]) + 1);
            }
            map.put(arr[i], i);
            max = Math.max(max, i - j + 1);
        }
        return max;
    }

    //27有重复项数字的全排列
    boolean[] mark;

    public ArrayList<ArrayList<Integer>> t27_permuteUnique(int[] num) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        LinkedList<Integer> track = new LinkedList<>();
        mark = new boolean[num.length];
        Arrays.sort(num);
        backtrack(num, res, track);
        return res;
    }

    public void backtrack(int[] num, ArrayList<ArrayList<Integer>> res, LinkedList<Integer> track) {
        if (track.size() == num.length) {
            res.add(new ArrayList<Integer>(track));
            return;
        }
        for (int i = 0; i < num.length; i++) {
            if (mark[i] || i > 0 && num[i] == num[i - 1] && !mark[i - 1]) {
                continue;
            }
            track.add(num[i]);
            mark[i] = true;
            backtrack(num, res, track);
            track.removeLast();
            mark[i] = false;
        }
    }

    //28通配符匹配
    public boolean t28_isMatch(String s, String p) {
        int row = s.length();
        int col = p.length();
        boolean[][] dp = new boolean[row + 1][col + 1];
        dp[0][0] = true;
        for (int j = 1; j < col + 1; j++) {
            if (dp[0][j - 1]) {
                if (p.charAt(j - 1) == '*') {
                    dp[0][j] = true;
                } else {
                    break;
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (p.charAt(j) == s.charAt(i) || p.charAt(j) == '?') {
                    dp[i + 1][j + 1] = dp[i][j];
                } else if (p.charAt(j) == '*') {
                    dp[i + 1][j + 1] = dp[i][j] || dp[i + 1][j] || dp[i][j + 1];
                }
            }
        }

        return dp[row][col];
    }

    //29实现二叉树先序，中序和后序遍历
    public int[][] t29_threeOrders(TreeNode root) {
        // write code here
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        List<Integer> list3 = new ArrayList<>();

        preOrder(root, list1);
        inOrder(root, list2);
        postOrder(root, list3);

        int[][] res = new int[3][list1.size()];
        for (int i = 0; i < list1.size(); i++) {
            res[0][i] = list1.get(i);
            res[1][i] = list2.get(i);
            res[2][i] = list3.get(i);
        }

        return res;
    }

    public void preOrder(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        list.add(root.val);
        preOrder(root.left, list);
        preOrder(root.right, list);
    }

    public void inOrder(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        inOrder(root.left, list);
        list.add(root.val);
        inOrder(root.right, list);
    }

    public void postOrder(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        postOrder(root.left, list);
        postOrder(root.right, list);
        list.add(root.val);
    }

    //30加起来和为目标值的组合(二)
    public ArrayList<ArrayList<Integer>> t30_combinationSum2(int[] num, int target) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> arr = new ArrayList<Integer>();
        if (num == null || num.length == 0 || target < 0) {
            return res;
        }
        Arrays.sort(num);
        dfs(num, target, res, arr, 0);
        return res;
    }

    void dfs(int[] num, int target, ArrayList<ArrayList<Integer>> res, ArrayList<Integer> arr, int start) {
        if (target == 0) {
            res.add(new ArrayList<Integer>(arr));
            return;
        }
        if (start >= num.length) {
            return;
        }
        for (int i = start; i < num.length; i++) {
            if (i > start && num[i] == num[i - 1]) {
                continue;
            }
            if (num[i] <= target) {
                arr.add(num[i]);
                dfs(num, target - num[i], res, arr, i + 1);
                arr.remove(arr.size() - 1);
            }
        }
        return;
    }

    //31最长的括号子串
    public int t31_longestValidParentheses(String s) {
        // write code here
        int maxans = 0;
        int[] dp = new int[s.length()];
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxans = Math.max(maxans, dp[i]);
            }
        }
        return maxans;
    }

    //32链表中的节点每k个一组翻转
    public ListNode t32_reverseKGroup(ListNode head, int k) {
        // write code here
        if (k <= 1) {
            return head;
        }
        if (head == null) {
            return head;
        }
        ListNode node = head;
        int len = length(head);
        head = node;
        int sx = len / k;
        ListNode result = new ListNode(0);
        ListNode now = result;
        int cnt = 0;
        for (int i = 0; i < sx; i++) {
            ListNode tmp = null;
            for (int j = 0; j < k; j++) {
                ListNode bl = head.next;
                head.next = tmp;
                tmp = head;
                head = bl;
            }
            now.next = tmp;
            while (now.next != null) {
                now = now.next;
            }
        }
        now.next = head;
        return result.next;
    }

    public int length(ListNode now) {
        int cnt = 0;
        if (now != null) {
            cnt = 1;
        }
        while (now.next != null) {
            cnt++;
            now = now.next;
        }
        return cnt;
    }

    //33合并k个已排序的链表
    public ListNode t33_mergeKLists(ArrayList<ListNode> lists) {
        return mergeList(lists, 0, lists.size() - 1);
    }

    public ListNode mergeList(ArrayList<ListNode> lists, int left, int right) {
        if (left == right) {
            return lists.get(left);
        }
        if (left > right) {
            return null;
        }
        int mid = left + ((right - left) >> 1);
        return merge(mergeList(lists, left, mid), mergeList(lists, mid + 1, right));
    }

    public ListNode merge(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null) {
            return l1 == null ? l2 : l1;
        }

        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        cur.next = (l1 == null ? l2 : l1);
        return dummy.next;
    }

    //34有效括号序列
    public boolean t34_isValid(String s) {
        // write code here
        Stack<Character> stk = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c == '(') {
                stk.push(')');
            } else if (c == '[') {
                stk.push(']');
            } else if (c == '{') {
                stk.push('}');
            } else {
                if (stk.isEmpty() || c != stk.pop()) {
                    return false;
                }
            }
        }
        return stk.isEmpty();
    }

    //35删除链表的倒数第n个节点
    public ListNode t55_removeNthFromEnd(ListNode head, int n) {
        // write code here
        int length = 0;
        ListNode p = head;
        ListNode q = head;
        while (head != null) {
            length++;
            head = head.next;
        }
        if (length < 2) {
            return null;
        }
        if (n == length) {
            return q.next;
        }
        int i = 0;
        while (p != null) {
            i++;
            if (i == length - n) {
                p.next = p.next.next;
            }
            p = p.next;
        }
        return q;
    }

    //36三数之和
    public ArrayList<ArrayList<Integer>> t36_threeSum(int[] num) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();

        if (num == null || num.length < 3) {
            return res;
        }
        Arrays.sort(num);// 排序
        for (int i = 0; i < num.length - 2; i++) {
            if (num[i] > 0) {
                break;// 如果当前数字大于0，则三数之和一定大于0，所以结束循环
            }
            if (i > 0 && num[i] == num[i - 1]) {
                continue;// 去重
            }
            int L = i + 1;
            int R = num.length - 1;

            while (L < R) {
                int sum = num[i] + num[L] + num[R];
                if (sum == 0) {
                    ArrayList<Integer> list = new ArrayList<>();
                    list.add(num[i]);
                    list.add(num[L]);
                    list.add(num[R]);
                    res.add(list);

                    while (L < R && num[L] == num[L + 1]) {
                        L++;
                    }
                    while (L < R && num[R] == num[R - 1]) {
                        R--;
                    }
                    L++;
                    R--;
                } else if (sum > 0) {
                    R--;
                } else if (sum < 0) {
                    L++;
                }
            }
        }
        return res;
    }

    //37最长公共前缀
    public String t37_longestCommonPrefix(String[] strs) {
        // //纵向扫描
        if (strs.length == 0 || strs == null) {
            return "";
        }

        int rows = strs.length;
        int cols = strs[0].length();
        //开始扫描
        for (int i = 0; i < cols; i++) {
            char firstChar = strs[0].charAt(i);
            for (int j = 1; j < rows; j++) {
                if (strs[j].length() == i || strs[j].charAt(i) != firstChar) {
                    return strs[0].substring(0, i);
                }
            }
        }
        return strs[0];
    }


    //38反转数字
    public int t38_reverse(int x) {
        int res = 0;
        while (x != 0) {
            int t = x % 10;
            int newRes = res * 10 + t;
            //如果数字溢出，直接返回0
            if ((newRes - t) / 10 != res)
                return 0;
            res = newRes;
            x = x / 10;
        }
        return res;
    }

    //39判断一棵二叉树是否为搜索二叉树和完全二叉树
    long pre = Long.MIN_VALUE;

    public boolean[] t39_judgeIt(TreeNode root) {
        // write code here
        return new boolean[]{isSBT(root), isCBT(root)};
    }

    public boolean isSBT(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (!isSBT(root.left)) {
            return false;
        }
        if (root.val <= pre) {
            return false;
        }
        pre = root.val;
        return isSBT(root.right);
    }

    public boolean isCBT(TreeNode root) {
        if (root == null) {
            return true;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean leaf = false;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (leaf && (node.left != null || node.right != null)) {
                    return false;
                }
                if (node.left == null && node.right != null) {
                    return false;
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                } else {
                    leaf = true;
                }
            }
        }
        return true;

    }

    //40两数之和
    public int[] t40_twoSum(int[] numbers, int target) {
        // write code here
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(target - numbers[i]))
                return new int[]{map.get(target - numbers[i]) + 1, i + 1};
            else
                map.put(numbers[i], i);
        }
        throw new IllegalArgumentException("No solution");
    }

    //41判断是不是平衡二叉树
    public int depth(TreeNode root) {
        if (root == null) return 0;
        int left = depth(root.left);
        if (left == -1) return -1;
        int right = depth(root.right);
        if (right == -1) return -1;
        if (left - right < (-1) || left - right > 1)
            return -1;
        else
            return 1 + (left > right ? left : right);
    }

    public boolean t41_IsBalanced_Solution(TreeNode root) {
        return depth(root) != -1;
    }

    //42扑克牌顺子
    public boolean t42_IsContinuous(int[] numbers) {
        int[] d = new int[14];
        d[0] = -5;
        int len = numbers.length;
        int max = -1;
        int min = 14;
        for (int i = 0; i < len; i++) {
            d[numbers[i]]++;
            if (numbers[i] == 0) {
                continue;
            }
            if (d[numbers[i]] > 1) {
                return false;
            }
            if (numbers[i] > max) {
                max = numbers[i];
            }
            if (numbers[i] < min) {
                min = numbers[i];
            }

        }
        if (max - min < 5) {
            return true;
        }
        return false;
    }

    //43斐波那契数列
    public int t43_Fibonacci(int n) {
        int[] ans = new int[50];
        ans[0] = 0;
        ans[1] = 1;
        for (int i = 2; i <= n; i++) {
            ans[i] = ans[i - 1] + ans[i - 2];
        }

        return ans[n];
    }

    //44两个链表的第一个公共结点
    public ListNode t44_FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        ListNode l1 = pHead1, l2 = pHead2;
        while (l1 != l2) {
            l1 = (l1 == null) ? pHead2 : l1.next;
            l2 = (l2 == null) ? pHead1 : l2.next;
        }
        return l1;
    }

    //45跳台阶
    public int t45_jumpFloor(int target) {
        if (target < 2) {
            return target;
        }

        int p = 0, q = 1, r = 2;
        for (int i = 3; i <= target; i++) {
            p = q;
            q = r;
            r = p + q;
        }

        return r;
    }

    //46单链表的排序
    public ListNode t46_sortInList(ListNode head) {
        // write code here
        if (head == null || head.next == null)
            return head;
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode newList = slow.next;
        slow.next = null;
        ListNode left = t46_sortInList(head);
        ListNode right = t46_sortInList(newList);

        ListNode lhead = new ListNode(-1);
        ListNode res = lhead;
        while (left != null && right != null) {
            if (left.val < right.val) {
                lhead.next = left;
                left = left.next;
            } else {
                lhead.next = right;
                right = right.next;
            }
            lhead = lhead.next;
        }
        lhead.next = left != null ? left : right;
        return res.next;
    }

    //47二叉树的镜像
    public TreeNode t47_Mirror(TreeNode pRoot) {
        if (pRoot == null)
            return pRoot;
        if (pRoot.left == null && pRoot.right == null)
            return pRoot;
        TreeNode temp = pRoot.left;
        pRoot.left = pRoot.right;
        pRoot.right = temp;
        t47_Mirror(pRoot.left);
        t47_Mirror(pRoot.right);
        return pRoot;
    }

    //48数组中出现次数超过一半的数字
    public int t48_MoreThanHalfNum_Solution(int[] array) {
        if (array == null || array.length == 0) return 0;
        int preValue = array[0];
        int count = 1;
        for (int i = 1; i < array.length; i++) {
            if (array[i] == preValue)
                count++;
            else {
                count--;
                if (count == 0) {
                    preValue = array[i];
                    count = 1;
                }
            }
        }
        int num = 0;
        for (int i = 0; i < array.length; i++)
            if (array[i] == preValue)
                num++;
        return (num > array.length / 2) ? preValue : 0;

    }

    //49数字在升序数组中出现的次数
    public int t49_GetNumberOfK(int[] array, int k) {
        int index = Arrays.binarySearch(array, k);
        if (index < 0) return 0;
        int cnt = 1;
        for (int i = index + 1; i < array.length && array[i] == k; i++)
            cnt++;
        for (int i = index - 1; i >= 0 && array[i] == k; i--)
            cnt++;
        return cnt;

    }

    //50用两个栈实现队列
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void t50_push(int node) {
        stack1.push(node);
    }

    public int pop() {
        if (stack2.size() <= 0) {
            while (stack1.size() != 0) {
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop();
    }

    //51反转链表
    public ListNode t51_ReverseList(ListNode head) {
        Stack<ListNode> stack = new Stack<>();
        //把链表节点全部摘掉放到栈中
        while (head != null) {
            stack.push(head);
            head = head.next;
        }
        if (stack.isEmpty())
            return null;
        ListNode node = stack.pop();
        ListNode dummy = node;
        //栈中的结点全部出栈，然后重新连成一个新的链表
        while (!stack.isEmpty()) {
            ListNode tempNode = stack.pop();
            node.next = tempNode;
            node = node.next;
        }
        //最后一个结点就是反转前的头结点，一定要让他的next
        //等于空，否则会构成环
        node.next = null;
        return dummy;
    }

    //52滑动窗口的最大值
    public ArrayList<Integer> t52_maxInWindows(int[] num, int size) {
        ArrayList<Integer> list = new ArrayList<Integer>();
        int max = 0;
        if (num.length == 0 || size > num.length || size == 0) {
            return list;
        }
        for (int i = 0; i <= num.length - size; i++) {
            max = num[i];
            for (int j = i; j < size + i; j++) {
                if (max < num[j]) {
                    max = num[j];
                }
            }
            list.add(max);
        }
        return list;
    }

    //53矩阵元素查找
    public int[] t53_findElement(int[][] mat, int n, int m, int x) {
        // write code here
        int nn = n - 1;
        int mm = 0;
        while (nn >= 0 && mm <= m - 1) {

            if (mat[nn][mm] == x)
                return new int[]{nn, mm};
            else if (mat[nn][mm] > x)
                nn--;
            else
                mm++;
        }
        return new int[]{};
    }

    //54字符串变形
    public String t54_trans(String s, int n) {
        StringBuffer ss = new StringBuffer();
        StringBuffer str = new StringBuffer();
        for (int i = n - 1; i >= 0; i--) {
            if (s.charAt(i) == ' ') {
                ss.append(str.toString() + " ");
                str = new StringBuffer();
            } else {
                //大小写反转
                char sss = (char) (s.charAt(i) < 97 ? s.charAt(i) + 32 : s.charAt(i) - 32);
                //将字符添加到str的首个位置保证单词不被反转
                str.insert(0, sss);
            }
        }
        return ss.append(str.toString()).toString();
    }

    //55最长上升子序列(三)
    public static int[] t55_LIS(int[] arr) {
        // write code here
        List<Integer> result = new ArrayList<>();
        int[] maxLength = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            if (result.size() > 0) {
                if (result.get(result.size() - 1) < arr[i]) {
                    result.add(arr[i]);
                    maxLength[i] = result.size();
                } else {
                    for (int j = result.size() - 1; j >= 0; j--) {
                        if (result.get(j) < arr[i]) {
                            result.set(j + 1, arr[i]);
                            maxLength[i] = j + 2;
                            break;
                        }
                        if (j == 0) {
                            result.set(0, arr[i]);
                            maxLength[i] = 1;
                        }
                    }
                }
            } else {
                result.add(arr[i]);
                maxLength[i] = 1;
            }
        }
        int[] resultArray = new int[result.size()];

        for (int i = arr.length - 1, j = result.size(); j > 0; i--) {
            if (maxLength[i] == j) {
                resultArray[--j] = arr[i];
            }
        }
        return resultArray;
    }

    //56最长公共子序列(二)
    public String t56_LCS(String s1, String s2) {
        int len1 = s1.length();
        int len2 = s2.length();
        if (len1 == 0 || len2 == 0)
            return "-1";
        int[][] dp = new int[len1 + 1][len2 + 1];
        for (int i = 0; i < len1 + 1; i++) {
            for (int j = 0; j < len2 + 1; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 0;
                    continue;
                }
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        int s1L = len1, s2L = len2;
        while (s1L != 0 && s2L != 0) {
            if (s1.charAt(s1L - 1) == s2.charAt(s2L - 1)) {
                sb.append(s1.charAt(s1L - 1));
                s1L--;
                s2L--;
            } else {
                if (dp[s1L - 1][s2L] > dp[s1L][s2L - 1]) {
                    s1L--;
                } else {
                    s2L--;
                }
            }
        }
        if (sb.length() == 0)
            return "-1";
        return sb.reverse().toString();
    }

    //57设计LRU缓存结构
    static class Node {
        int key, value;
        Node prev, next;

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    private Map<Integer, Node> map = new HashMap<>();
    private Node head = new Node(-1, -1);
    private Node tail = new Node(-1, -1);
    private int k;

    public int[] t57_LRU(int[][] operators, int k) {
        // write code here
        this.k = k;
        head.next = tail;
        tail.prev = head;
        int len = (int) Arrays.stream(operators).filter(x -> x[0] == 2).count();
        int[] ans = new int[len];
        int cnt = 0;
        for (int i = 0; i < operators.length; i++) {
            if (operators[i][0] == 1) {
                set(operators[i][1], operators[i][2]);
            } else {
                ans[cnt++] = get(operators[i][1]);
            }
        }
        return ans;
    }

    public void set(int key, int value) {
        if (get(key) > -1) {
            map.get(key).value = value;
        } else {
            if (map.size() == k) {
                int rk = tail.prev.key;
                tail.prev.prev.next = tail;
                tail.prev = tail.prev.prev;
                map.remove(rk);
            }
            Node node = new Node(key, value);
            map.put(key, node);
            removeToHead(node);
        }
    }

    public int get(int key) {
        if (map.containsKey(key)) {
            Node node = map.get(key);
            node.prev.next = node.next;
            node.next.prev = node.prev;

            removeToHead(node);
            return node.value;
        }
        return -1;
    }

    public void removeToHead(Node node) {
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
        node.prev = head;
    }

    //58设计LFU缓存结构
    public int[] t58_LFU(int[][] operators, int k) {
        HashMap<Integer, Nodex> count = new HashMap<>();
        HashMap<Integer, int[]> temp = new HashMap<>();
        PriorityQueue<Integer> minHeap = new PriorityQueue<>((o1, o2) -> count.get(o1).count.equals(count.get(o2).count) ?
                count.get(o1).time.compareTo(count.get(o2).time) : count.get(o1).count.compareTo(count.get(o2).count));
        ArrayList<Integer> res = new ArrayList<>();
        int time = 0;
        for (int i = 0; i < operators.length; i++) {
            if (operators[i][0] == 1) {
                if (temp.size() == k) {
                    int poll = minHeap.poll();
                    count.remove(poll);
                    temp.remove(poll);
                }
                temp.put(operators[i][1], operators[i]);
                if (count.containsKey(operators[i][1])) {
                    Nodex node = new Nodex(count.get(operators[i][1]).count + 1, time++);
                    count.put(operators[i][1], node);
                    minHeap.remove(operators[i][1]);
                    minHeap.offer(operators[i][1]);
                } else {
                    Nodex node1 = new Nodex(1, time++);
                    count.put(operators[i][1], node1);
                    minHeap.offer(operators[i][1]);
                }
            } else if (operators[i][0] == 2) {
                if (temp.get(operators[i][1]) == null) {
                    res.add(-1);
                } else {
                    int[] operator = temp.get(operators[i][1]);
                    res.add(operator[2]);
                    if (count.containsKey(operators[i][1])) {
                        Nodex node2 = new Nodex(count.get(operators[i][1]).count + 1, time++);
                        count.put(operators[i][1], node2);
                        minHeap.remove(operators[i][1]);
                        minHeap.offer(operators[i][1]);
                    } else {
                        Nodex node3 = new Nodex(1, time++);
                        count.put(operators[i][1], node3);
                        minHeap.offer(operators[i][1]);
                    }

                }
            }
        }
        int[] result = new int[res.size()];
        for (int i = 0; i < res.size(); i++) {
            result[i] = res.get(i);
        }
        return result;
    }


    class Nodex {
        public Integer count;
        public Integer time;

        public Nodex(Integer count, Integer time) {
            this.count = count;
            this.time = time;
        }
    }

    //59数组中的最长连续子序列
    public int t59_MLS(int[] arr) {
        Set<Integer> set = new HashSet<>();
        for (int num : arr)
            set.add(num);
        int longest = 0;
        for (int num : arr) {
            if (set.contains(num - 1))
                continue;
            int currentNum = num;
            int count = 1;
            while (set.contains(currentNum + 1)) {
                currentNum++;
                count++;
            }
            longest = Math.max(longest, count);
        }
        return longest;
    }

    //60判断一个链表是否为回文结构
    public class ListNodex {
        int val;
        ListNodex next = null;
    }

    public boolean t60_isPail(ListNodex head) {
        ListNodex temp = head;
        Stack<Integer> stack = new Stack();
        while (temp != null) {
            stack.push(temp.val);
            temp = temp.next;
        }

        while (head != null) {
            if (head.val != stack.pop()) {
                return false;
            }
            head = head.next;
        }
        return true;
    }

    //61判断t1树中是否有与t2树完全相同的子树
    public boolean t61_isContains(TreeNode root1, TreeNode root2) {
        if (root1 == null)
            return false;

        return t61_isContains(root1.left, root2) || t61_isContains(root1.right, root2)
                || check(root1, root2);
    }

    public boolean check(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null)
            return true;
        if (root1 == null || root2 == null || root1.val != root2.val)
            return false;
        return check(root1.left, root2.left) && check(root1.right, root2.right);
    }

    //62把字符串转换成整数(atoi)
    public int t62_StrToInt(String s) {
        int len = s.length();
        if (len == 0) return 0;
        int sign = 1;//默认为正数
        long num = 0;
        int i = 0;
        while (i < len && s.charAt(i) == ' ') i++;

        if (i < len) {
            if (s.charAt(i) == '-') {
                sign = -1;
                i++;
            } else if (s.charAt(i) == '+') i++;
        }
        while (i < len) {
            if (s.charAt(i) >= '0' && s.charAt(i) <= '9') {
                num = num * 10 + (s.charAt(i) - '0');
                if (sign == -1 && num * (-1) < Integer.MIN_VALUE) return Integer.MIN_VALUE;
                else if (sign == 1 && num > Integer.MAX_VALUE) return Integer.MAX_VALUE;
                i++;
            } else break;
        }
        int res = (int) num;
        res *= sign;
        return res;
    }

    //63反转字符串
    public String t63_solve(String str) {
        char[] cstr = str.toCharArray();
        int len = str.length();
        for (int i = 0; i < len / 2; i++) {
            char t = cstr[i];
            cstr[i] = cstr[len - 1 - i];
            cstr[len - 1 - i] = t;
        }
        return new String(cstr);
    }

    //64二分查找-II
    public int t64_search(int[] nums, int target) {
        //边界条件判断
        if (nums == null || nums.length == 0)
            return -1;
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left] == target ? left : -1;
    }

    //65三个数的最大乘积

    public long t65_solve(int[] A) {
        // write code here
        int max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE, max3 = Integer.MIN_VALUE;
        int min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        for (int num : A) {
            if (num > max1) {
                max3 = max2;
                max2 = max1;
                max1 = num;
            } else if (num > max2) {
                max3 = max2;
                max2 = num;
            } else if (num > max3) max3 = num;
            if (num < min1) {
                min2 = min1;
                min1 = num;
            } else if (num < min2) min2 = num;
        }
        return Math.max((long) max1 * max2 * max3, (long) max1 * min1 * min2);
    }


    //66寻找峰值
    public int t66_findPeakElement(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return right;
    }

    //67最大数
    public String t67_solve(int[] nums) {
        // write code here
        ArrayList<String> list = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            list.add(String.valueOf(nums[i]));
        }
        Collections.sort(list, new Comparator<String>() {
            @Override
            public int compare(String a, String b) {
                return (b + a).compareTo(a + b);
            }
        });
        if (list.get(0).equals("0")) return "0";

        StringBuilder res = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            res.append(list.get(i));
        }
        return res.toString();


    }

    //68验证IP地址
    public String t68_solve(String IP) {
        return validIPv4(IP) ? "IPv4" : (validIPv6(IP) ? "IPv6" : "Neither");
    }

    private boolean validIPv4(String IP) {
        String[] strs = IP.split("\\.", -1);
        if (strs.length != 4) {
            return false;
        }

        for (String str : strs) {
            if (str.length() > 1 && str.startsWith("0")) {
                return false;
            }
            try {
                int val = Integer.parseInt(str);
                if (!(val >= 0 && val <= 255)) {
                    return false;
                }
            } catch (NumberFormatException numberFormatException) {
                return false;
            }
        }
        return true;
    }

    private boolean validIPv6(String IP) {
        String[] strs = IP.split(":", -1);
        if (strs.length != 8) {
            return false;
        }

        for (String str : strs) {
            if (str.length() > 4 || str.length() == 0) {
                return false;
            }
            try {
                int val = Integer.parseInt(str, 16);
            } catch (NumberFormatException numberFormatException) {
                return false;
            }
        }
        return true;
    }

    //69把数字翻译成字符串
    public int t69_solve(String nums) {
        if (nums.length() == 0 || nums.charAt(0) == '0')
            return 0;
        int[] dp = new int[nums.length()];
        dp[0] = 1;
        for (int i = 1; i < dp.length; i++) {
            if (nums.charAt(i) != '0') {
                dp[i] = dp[i - 1];
            }
            //  3 2 4
            int num = (nums.charAt(i - 1) - '0') * 10 + (nums.charAt(i) - '0');
            if (num >= 10 && num <= 26) {
                if (i == 1) {
                    dp[i] += 1;
                } else {
                    dp[i] += dp[i - 2];
                }
            }
        }
        return dp[nums.length() - 1];

    }

    //70合并二叉树
    public TreeNode t70_mergeTrees(TreeNode t1, TreeNode t2) {
        if (t1 == null) return t2;
        if (t2 == null) return t1;
        t1.val += t2.val;
        t1.left = t70_mergeTrees(t1.left, t2.left);
        t1.right = t70_mergeTrees(t1.right, t2.right);
        return t1;
    }

    //71最小的K个数
    public ArrayList<Integer> t71_GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> list = new ArrayList<Integer>();
        if (input == null || k > input.length || k <= 0) return list;
        PriorityQueue<Integer> queue = new PriorityQueue<Integer>(new Comparator<Integer>() {
            public int compare(Integer i1, Integer i2) {
                return i2.compareTo(i1);
            }
        });
        int len = input.length;
        for (int i = 0; i < len; ++i) {
            if (queue.size() != k) {
                queue.offer(input[i]);
            } else if (queue.peek() > input[i]) {
                queue.poll();
                queue.offer(input[i]);
            }
        }
        Iterator<Integer> it = queue.iterator();
        while (it.hasNext()) {
            list.add(it.next());
        }
        return list;
    }

    //72字符串的排列
    HashSet<String> res = new HashSet<String>();
    ArrayList<String> r = new ArrayList<String>();
    int length;
    char[] array;

    public ArrayList<String> t72_Permutation(String str) {
        if (str.length() == 0) return new ArrayList<String>();
        init(str);
        perm(0);
        r = new ArrayList<String>(res);
        Collections.sort(r);
        return r;
    }

    public void init(String str) {
        length = str.length();
        array = str.toCharArray();
        Arrays.sort(array);
        res = new HashSet<String>();
    }

    public void perm(int k) {
        if (k == length) {
            String s = new String(array);
            if (!res.contains(s)) res.add(s);
        } else {
            int j = k;
            for (; j < length; j++) {
                swap(k, j);
                perm(k + 1);
                swap(k, j);
            }
        }
    }

    public void swap(int i, int j) {
        char tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }

    //73最长公共子串
    public String t73_LCS(String str1, String str2) {
        int maxLenth = 0;
        int maxLastIndex = 0;
        int[][] dp = new int[str1.length() + 1][str2.length() + 1];
        for (int i = 0; i < str1.length(); i++) {
            for (int j = 0; j < str2.length(); j++) {
                if (str1.charAt(i) == str2.charAt(j)) {
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                    if (dp[i + 1][j + 1] > maxLenth) {
                        maxLenth = dp[i + 1][j + 1];
                        maxLastIndex = i;
                    }
                } else {
                    dp[i + 1][j + 1] = 0;
                }
            }
        }
        return str1.substring(maxLastIndex - maxLenth + 1, maxLastIndex + 1);
    }

    //74接雨水问题
    public static long t74_maxWater(int[] arr) {
        if (arr == null || arr.length <= 2) {
            return 0;
        }
        int left = 0, right = arr.length - 1;
        long sum = 0;
        int mark = Math.min(arr[left], arr[right]);
        while (left < right) {
            if (arr[left] < arr[right]) {
                left++;
                if (arr[left] < mark) {
                    sum += mark - arr[left];
                } else {
                    mark = Math.min(arr[left], arr[right]);
                }
            } else {
                right--;
                if (arr[right] < mark) {
                    sum += mark - arr[right];
                } else {
                    mark = Math.min(arr[right], arr[left]);
                }
            }
        }
        return sum;
    }

    //75环形链表的约瑟夫问题
    public int t75_ysf(int n, int m) {
        // write code here
        ListNode head = new ListNode(1);
        ListNode tail = head;
        for (int i = 2; i <= n; i++) {
            tail.next = new ListNode(i);
            tail = tail.next;
        }
        tail.next = head;
        ListNode index = head;
        ListNode pre = tail;
        int k = 0;
        while (index.next != null && index.next != index) {
            k++;
            ListNode next = index.next;
            if (k == m) {
                pre.next = pre.next.next;
                k = 0;
            }
            pre = index;
            index = next;
        }
        return index.val;
    }

    //76链表的奇偶重排
    public ListNode t76_oddEvenList(ListNode head) {
        if (head == null) {
            return head;
        }
        ListNode evenHead = head.next;
        ListNode odd = head, even = evenHead;
        while (even != null && even.next != null) {
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }

    //77买卖股票的最好时机(二)
    public int t77_maxProfit(int[] prices) {
        int ans = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                ans += prices[i] - prices[i - 1];
            }
        }
        return ans;
    }

    //78排序
    public int[] t78_MySort(int[] arr) {
        quickSort(arr, 0, arr.length - 1);
        return arr;
    }

    public void quickSort(int[] list, int left, int right) {
        if (left < right) {
            int point = partition(list, left, right);
            quickSort(list, left, point - 1);
            quickSort(list, point + 1, right);
        }
    }

    public int partition(int[] list, int left, int right) {
        int first = list[left];
        while (left < right) {
            while (left < right && list[right] >= first) {
                right--;
            }

            swap(list, left, right);

            while (left < right && list[left] <= first) {
                left++;
            }

            swap(list, left, right);
        }
        return left;
    }

    private void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    //79判断是否为回文字符串
    public boolean t79_judge(String str) {
        if (str.length() == 0)
            return true;
        //两个指针，一个从左边开始，一个从右边开始，每次两个
        //指针都同时往中间挪，只要两个指针指向的字符不一样就返回false
        int left = 0;
        int right = str.length() - 1;
        while (left < right) {
            if (str.charAt(left++) != str.charAt(right--))
                return false;
        }
        return true;
    }

    //80数组中只出现一次的数（其它数出现k次）
    public int t80_foundOnceNumber(int[] arr, int k) {
        // write code here
        HashMap<Integer, Boolean> map = new HashMap<>();
        for (int num : arr) {
            if (map.containsKey(num)) {
                map.put(num, true);
            } else {
                map.put(num, false);
            }
        }
        Set<Integer> set = map.keySet();
        for (int num : set) {
            if (map.get(num) == false) {
                return num;
            }
        }
        return 0;
    }

    //冒泡排序
    public int[] bubbleSort(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < nums.length - i - 1; j++) {
                if (nums[j] > nums[j + 1]) {
                    int temp = nums[j];
                    nums[j] = nums[j + 1];
                    nums[j + 1] = temp;
                }
            }
        }
        return nums;
    }

}